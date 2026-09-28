// BipedalWalker-v3 como entorno de RLTools, sin Box2D (física en physics/).
// Adaptado de Gymnasium (MIT, Oleg Klimov / Andrea Pierré); ver physics/ para la parte de Box2D.
//
// Un `Parameters` guarda lo que es común a un conjunto de episodios y se comparte de solo lectura
// entre individuos (geometría del caminante y terreno, ~1.1 KB en float); el `State` es solo lo que
// cambia paso a paso (~200 B).
#pragma once

#include <rl_tools/rl/environments/environments.h>

#include "physics/raycast.h"
#include "physics/reset.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::bipedal_walker {

    template <typename T, typename TI>
    struct Parameters {
        static constexpr T DT = static_cast<T>(1.0 / 50.0);
        // Iteraciones del solver de Box2D: 180/60 son las del entorno original. Bajarlas reduce
        // el coste por paso a costa de fidelidad (ver la calibración en tests/benchmarks).
        static constexpr TI VELOCITY_ITERATIONS = 6 * 30;
        static constexpr TI POSITION_ITERATIONS = 2 * 30;

        ::bipedal_walker::physics::Geometry<T> geometry;
        ::bipedal_walker::physics::Terrain<T> terrain;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = Parameters<T_T, T_TI>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    template <typename T, typename TI>
    struct State : ::bipedal_walker::physics::WalkerState<T> {
        static constexpr TI DIM = 5 * 6;  // 5 cuerpos x (x, y, ángulo, vx, vy, w)
    };

    // Observación de Gymnasium: 14 valores de estado y 10 rayos del lidar.
    template <typename TI>
    struct Observation {
        static constexpr TI DIM = 24;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments {

    template <typename T_SPEC>
    struct BipedalWalker : Environment<typename T_SPEC::T, typename T_SPEC::TI> {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = bipedal_walker::State<T, TI>;
        using Parameters = typename SPEC::PARAMETERS;
        using Observation = bipedal_walker::Observation<TI>;
        using ObservationPrivileged = Observation;
        static constexpr TI ACTION_DIM = 4;  // torque de cadera y rodilla de cada pierna, en [-1, 1]
        static constexpr TI EPISODE_STEP_LIMIT = 1600;
    };

    // BipedalWalkerHardcore-v3: mismo caminante y física, terreno con fosos/tocones/escaleras
    // (sample_initial_parameters se especializa para este tipo en operations_generic.h) y límite
    // de episodio más largo, como en Gymnasium.
    template <typename T_SPEC>
    struct BipedalWalkerHardcore : BipedalWalker<T_SPEC> {
        using TI = typename T_SPEC::TI;
        static constexpr TI EPISODE_STEP_LIMIT = 2000;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END
