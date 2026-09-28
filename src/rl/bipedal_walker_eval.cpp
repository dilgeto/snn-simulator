// Evaluación de una SNN sobre BipedalWalker-v3 (port a C++, sin Box2D).
//
// Sigue el mismo patrón que pendulum_eval.cpp y acrobot_eval.cpp: red fija, un decodificador de
// tasa de disparo y un barrido de pesos sinápticos compartidos. A diferencia de esos, aquí la
// topología (24 entradas, 4 salidas) y la observación son mucho más grandes, así que sirve además
// de banco de pruebas de coste: cada paso de entorno dispara una ventana de simulación de SNN.
//
// Para el estudio de fidelidad por ranking (ver tests/rl/bipedal_walker/ranking_eval.cpp y
// scripts/bipedal_reference/analyze_ranking.py) conviene sustituir esta red fija por la población
// evolucionada real; esa es la pieza que aún falta para repetir el estudio con controladores no
// caóticos.
#include <rl_tools/operations/cpu.h>
#include "../../include/core/network.hpp"
#include "../../include/core/simulator.hpp"
#include "../../include/decoding/rlDecoder.hpp"
#include "../../include/rl/environments/bipedal_walker/operations_generic.h"
#include <iostream>
#include <vector>

namespace rlt = rl_tools;
namespace phys = bipedal_walker::physics;

using T      = double;
using TI     = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using EnvSpec = rlt::rl::environments::bipedal_walker::Specification<T, TI>;
using Env    = rlt::rl::environments::BipedalWalker<EnvSpec>;
using RNG    = rlt::devices::random::CPU::ENGINE<>;

using ActionMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>>;
using ObsMatrix    = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 24, false>>;

// Observación (24 dims) → corrientes de entrada [0, 20] mA.
//   obs[0..3]   hull: ángulo, vel. angular, vx, vy — normalizados, pero sin cota dura
//   obs[4,6,9,11]  ángulos de junta (cadera ~[-0.8,1.1], rodilla+1 ~[-0.6,1.9])
//   obs[5,7,10,12] velocidades de junta / velocidad nominal — puede pasar de ±1 con impactos
//   obs[8,13]   contacto de pierna, ya en {0,1}
//   obs[14..23] lidar, ya en [0,1]
// Los ocho primeros grupos se acotan a [-2,2] antes de reescalar: cubre el rango normal con
// margen para picos de impacto, sin que un valor atípico sature la corriente en un extremo fijo.
static std::vector<double> obsToCurrents(const ObsMatrix& obs) {
    std::vector<double> currents(24);
    for (int i = 0; i < 14; i++) {
        double clipped = std::min(2.0, std::max(-2.0, rlt::get(obs, 0, i)));
        currents[i] = (clipped + 2.0) * 5.0;  // [-2,2] -> [0,20]
    }
    for (int i = 14; i < 24; i++) {
        currents[i] = rlt::get(obs, 0, i) * 20.0;  // lidar ya en [0,1] -> [0,20]
    }
    return currents;
}

int main() {
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 42);

    Env env;
    Env::Parameters params;
    rlt::sample_initial_parameters(device, env, params, rng);

    // SNN: 24 entradas -> 12 ocultas -> 4 salidas, todo a todo entre capas contiguas.
    Network net(1.0);
    for (int i = 0; i < 24; i++) {
        net.addInputNeuron(NeuronType::REGULAR_SPIKING);
    }

    std::vector<int> hidden;
    for (int i = 0; i < 12; i++) {
        hidden.push_back(net.addHiddenNeuron(NeuronType::REGULAR_SPIKING));
    }

    std::vector<int> outputs;
    for (int i = 0; i < 4; i++) {
        outputs.push_back(net.addOutputNeuron(NeuronType::REGULAR_SPIKING));
    }

    for (int in_id : net.getInputNeuronIds()) {
        for (int h_id : hidden) {
            net.addSynapse(in_id, h_id);
        }
    }
    for (int h_id : hidden) {
        for (int out_id : outputs) {
            net.addSynapse(h_id, out_id);
        }
    }

    Simulator sim(&net);
    RLDecoder decoder(RLDecoder::DecodingType::RATE, 50.0);

    std::vector<double> test_weights = {5.0, 10.0, 20.0};

    for (double weight : test_weights) {
        Env::State state;
        rlt::sample_initial_state(device, env, params, state, rng);

        double total_reward = 0.0;
        TI steps_taken = 0;

        for (TI step = 0; step < Env::EPISODE_STEP_LIMIT; step++) {
            ObsMatrix obs_mat;
            Env::Observation obs_type;
            rlt::observe(device, env, params, state, obs_type, obs_mat, rng);
            auto currents = obsToCurrents(obs_mat);

            // Simular la SNN 50 ms con el peso sináptico compartido de esta corrida.
            auto result = sim.simulate(0, currents, weight, 50.0);

            // Tasa de disparo de cada salida → uno de los 4 torques, en [-1, 1].
            ActionMatrix action_mat;
            for (int k = 0; k < 4; k++) {
                double action_value = 0.0;
                if (!result.output_spike_times[k].empty()) {
                    double rate = decoder.decodeContinuousAction(result.output_spike_times[k]);
                    action_value = rate * 2.0 - 1.0;
                }
                rlt::set(action_mat, 0, k, action_value);
            }

            Env::State next_state;
            rlt::step(device, env, params, state, action_mat, next_state, rng);
            T reward_val = rlt::reward(device, env, params, state, action_mat, next_state, rng);

            total_reward += reward_val;
            state = next_state;
            steps_taken++;

            if (rlt::terminated(device, env, params, state, rng)) {
                break;
            }
        }

        T x = phys::body_origin(params.geometry, state.bodies, 0).x;
        std::cout << "Peso: "          << weight
                  << " | Pasos: "      << steps_taken
                  << " | Recompensa: " << total_reward
                  << " | Avance x: "   << x
                  << "\n";
    }

    return 0;
}
