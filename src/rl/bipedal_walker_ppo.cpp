// Entrena BipedalWalker (port a C++, sin Box2D) con el PPO genérico de RLTools.
//
// Sigue el patrón de external/rl-tools/src/rl/environments/pendulum/ppo/cpu/{config,training}.cpp:
// el bucle `rl::algorithms::ppo::loop::core` es genérico sobre cualquier entorno que cumpla la
// interfaz estándar (malloc/free/init/sample_initial_parameters/sample_initial_state/step/reward/
// observe/terminated + Observation::DIM/ObservationPrivileged::DIM/ACTION_DIM/EPISODE_STEP_LIMIT),
// que es exactamente lo que operations_generic.h implementa para BipedalWalker y
// BipedalWalkerHardcore. No hace falta ningún archivo "ppo" específico del entorno.
//
// Hiperparámetros para un entrenamiento real (no el smoke test de 200k pasos de la corrida
// anterior), calcados del ejemplo de Ant de RLTools (external/rl-tools/src/rl/environments/
// mujoco/ant/ppo/parameters.h), la tarea más parecida del propio repo: locomoción con patas,
// contactos y control continuo. 5M pasos es un presupuesto típico de la literatura para que
// BipedalWalker-v3 llegue a caminar bien (retorno ~300); con menos puede quedarse en "se para
// pero no avanza", que es donde termina el smoke test de 200k.
//
// Costo: con ACTOR/CRITIC_HIDDEN_DIM=256 y un dataset de 32768 muestras por iteración, el
// backend genérico de CPU (sin BLAS) va a ser bastante más lento que el smoke test (~5-6k
// pasos/s con hidden_dim=64). Para una corrida de este tamaño conviene compilar con un backend
// con BLAS (OpenBLAS o MKL, ver RL_TOOLS_BACKEND_ENABLE_OPENBLAS/MKL en cpu_mux.h) en vez del
// genérico.
#include <rl_tools/operations/cpu_mux.h>

#include "../../include/rl/environments/bipedal_walker/operations_generic.h"

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

// Para guardar el actor entrenado al final, como código C++ (pesos embebidos en arrays): el
// mismo formato que usa el paso de checkpoint de RLTools, pero escrito una sola vez al terminar,
// sin necesitar los módulos extrack/checkpoint del bucle (que arman su propia carpeta de
// experimento con nombre por timestamp).
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include <utility>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::template ENGINE<>;
static constexpr bool DYNAMIC_ALLOCATION = true;

// HARDCORE=1 en el compilador entrena BipedalWalkerHardcore en vez del modo normal.
#ifndef BIPEDAL_WALKER_HARDCORE
#define BIPEDAL_WALKER_HARDCORE 0
#endif

using ENV_SPEC = rlt::rl::environments::bipedal_walker::Specification<T, TI>;
#if BIPEDAL_WALKER_HARDCORE
using ENVIRONMENT = rlt::rl::environments::BipedalWalkerHardcore<ENV_SPEC>;
#else
using ENVIRONMENT = rlt::rl::environments::BipedalWalker<ENV_SPEC>;
#endif

struct LOOP_CORE_PARAMETERS : rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
    // Red más ancha que el smoke test (64): como Ant, el gradiente de dificultad de BipedalWalker
    // (límites de junta, contactos, lidar) se beneficia de más capacidad que un péndulo.
    static constexpr TI ACTOR_HIDDEN_DIM = 256;
    static constexpr TI CRITIC_HIDDEN_DIM = 256;
    // ACTOR_NUM_LAYERS/CRITIC_NUM_LAYERS quedan en 3 (el valor por defecto de la plantilla), igual
    // que en Ant.

    static constexpr TI N_ENVIRONMENTS = 64;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 512;  // dataset de 64*512 = 32768 por iteración
    static constexpr TI BATCH_SIZE = 4096;                     // 8 minilotes por época

    // Presupuesto total de pasos de entorno (todas las envs, todas las iteraciones). 5M es un
    // punto de partida razonable; bájalo para iterar más rápido, o súbelo a 10-20M si a los 5M no
    // llegó a un retorno positivo.
    static constexpr TI TOTAL_STEP_LIMIT = 5000000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT / (ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;

    using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
    using CRITIC_OPTIMIZER_PARAMETERS = ACTOR_OPTIMIZER_PARAMETERS;

    struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE> {
        // N_EPOCHS, GAMMA, ACTION_ENTROPY_COEFFICIENT, ADAPTIVE_LEARNING_RATE*: igual que el
        // ejemplo de Ant de RLTools. El resto (LAMBDA, EPSILON_CLIP, NORMALIZE_ADVANTAGE, ...)
        // se deja en el valor por defecto de la biblioteca, que ya es razonable para PPO continuo.
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0;
        static constexpr TI N_EPOCHS = 4;
        static constexpr T GAMMA = 0.99;
        static constexpr T INITIAL_ACTION_STD = 0.5;
        static constexpr bool NORMALIZE_OBSERVATIONS = true;
        // Ajusta la tasa de aprendizaje según la divergencia KL entre épocas: evita que una
        // actualización grande desestabilice la política en una corrida larga.
        static constexpr bool ADAPTIVE_LEARNING_RATE = true;
        static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;
    };
};

using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<
    TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS,
    rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;

template <typename NEXT>
struct LOOP_EVAL_PARAMETERS : rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, NEXT> {
    static constexpr TI EVALUATION_INTERVAL = 10;
    static constexpr TI NUM_EVALUATION_EPISODES = 32;  // más episodios: el retorno de BipedalWalker es ruidoso
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL + 1;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
using LOOP_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVALUATION_CONFIG>;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

int main(int argc, char** argv) {
    // Ruta del modelo guardado al final: argumento opcional, o el nombre por defecto.
    std::string output_path = argc > 1 ? argv[1]
                                        : (BIPEDAL_WALKER_HARDCORE ? "bipedal_walker_hardcore_actor.h" : "bipedal_walker_actor.h");

    std::cout << "BipedalWalker" << (BIPEDAL_WALKER_HARDCORE ? "Hardcore" : "") << " + PPO (RLTools)\n";
    std::cout << "  pasos totales objetivo: " << LOOP_CORE_PARAMETERS::TOTAL_STEP_LIMIT
              << " (" << LOOP_CORE_PARAMETERS::STEP_LIMIT << " iteraciones x "
              << LOOP_CORE_PARAMETERS::N_ENVIRONMENTS << " envs x "
              << LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV << " pasos/env)\n";
    std::cout << "  modelo final en: " << std::filesystem::absolute(output_path).string() << "\n";

    DEVICE device;
    rlt::log(device, LOOP_CONFIG{});

    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, 0);

    while (!rlt::step(device, ts)) {
    }

    rlt::log(device, device.logger, "PPO steps: ", ts.step);

    // Guarda el actor entrenado como código C++ (pesos embebidos), igual que el formato del
    // paso de checkpoint de RLTools pero en un único archivo, al final del entrenamiento.
    //
    // El actor de ts (rlt::get_actor) tiene capacidad Gradient: además de los pesos carga el
    // estado de Adam (momento 1 y 2) para poder seguir entrenando. Para un modelo final de
    // inferencia eso no sirve y solo infla el archivo (en una prueba corta ya salió >60 MB para
    // una red de 256 unidades); se copia a una variante Forward (solo pesos) antes de guardar,
    // igual que hace rl::loop::steps::checkpoint::save en RLTools.
    {
        auto& actor = rlt::get_actor(ts);
        using ACTOR_TYPE = rlt::utils::typing::remove_reference_t<decltype(actor)>;
        using EVALUATION_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
        EVALUATION_ACTOR_TYPE evaluation_actor;
        rlt::malloc(device, evaluation_actor);
        rlt::copy(device, device, actor, evaluation_actor);

        std::string code = rlt::save_code(device, evaluation_actor, std::string("bipedal_walker::actor"), true);
        std::ofstream out(output_path);
        if (!out) {
            std::cerr << "No se pudo abrir " << output_path << " para escribir el modelo\n";
        } else {
            out << code;
            std::cout << "Modelo guardado en " << std::filesystem::absolute(output_path).string() << "\n";
        }
        rlt::free(device, evaluation_actor);
    }

    rlt::free(device, ts);
    return 0;
}
