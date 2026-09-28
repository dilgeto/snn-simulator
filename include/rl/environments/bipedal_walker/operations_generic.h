// Operaciones de RLTools para el BipedalWalker sin Box2D.
#pragma once

#include <rl_tools/rl/environments/operations_generic.h>

#include "bipedal_walker.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::bipedal_walker {

    // Adaptador de la fuente de números aleatorios de RLTools a la interfaz del generador de
    // terreno (uniform en [low, high), integers en [low, high) como np_random).
    template <typename DEVICE, typename RNG>
    struct TerrainRng {
        RNG& rng;

        double uniform(double low, double high) {
            return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), low, high, rng);
        }
        int integers(int low, int high) {
            return random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), low, high - 1, rng);
        }
        double random() {
            return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), 0.0, 1.0, rng);
        }
    };

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T shaping_delta(const ::bipedal_walker::physics::Geometry<T>& g,
                                                const ::bipedal_walker::physics::BodyState<T> before[::bipedal_walker::physics::BODY_COUNT],
                                                const ::bipedal_walker::physics::BodyState<T> after[::bipedal_walker::physics::BODY_COUNT]) {
        namespace phys = ::bipedal_walker::physics;
        using C = phys::Constants<T>;
        // shaping = 130 * x / SCALE - 5 * |ángulo del casco|, con x la posición del origen del casco.
        // Se resta directamente la diferencia de x para no perder precisión en float.
        const T dx = phys::body_origin(g, after, 0).x - phys::body_origin(g, before, 0).x;
        return static_cast<T>(130) * dx / static_cast<T>(C::SCALE)
               - static_cast<T>(5) * (phys::abs(after[0].a) - phys::abs(before[0].a));
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env) {}
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env) {}
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env) {}

    // Parámetros por defecto: terreno plano a la altura inicial.
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_parameters(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                        typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters) {
        using T = typename SPEC::T;
        namespace phys = ::bipedal_walker::physics;
        parameters.geometry = phys::make_geometry<T>();
        for (int i = 0; i < phys::Constants<T>::TERRAIN_LENGTH; ++i) {
            parameters.terrain.y[i] = static_cast<T>(phys::Constants<T>::TERRAIN_HEIGHT);
        }
    }

    // Terreno aleatorio del modo normal, con las mismas llamadas al RNG que Gymnasium.
    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                               typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters, RNG& rng) {
        using T = typename SPEC::T;
        parameters.geometry = ::bipedal_walker::physics::make_geometry<T>();
        rl::environments::bipedal_walker::TerrainRng<DEVICE, RNG> terrain_rng{rng};
        ::bipedal_walker::physics::generate_terrain(parameters.terrain, terrain_rng);
    }

    // BipedalWalkerHardcore: terreno con fosos, tocones y escaleras. step/reward/observe/terminated
    // se reutilizan de BipedalWalker<SPEC> por deducción de plantilla desde la clase base (misma
    // física y misma fórmula de recompensa que el modo normal; solo cambian el terreno y
    // EPISODE_STEP_LIMIT, definido en bipedal_walker.h).
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_parameters(DEVICE& device, const rl::environments::BipedalWalkerHardcore<SPEC>& env,
                                                        typename rl::environments::BipedalWalkerHardcore<SPEC>::Parameters& parameters) {
        using T = typename SPEC::T;
        namespace phys = ::bipedal_walker::physics;
        parameters.geometry = phys::make_geometry<T>();
        for (int i = 0; i < phys::Constants<T>::TERRAIN_LENGTH; ++i) {
            parameters.terrain.y[i] = static_cast<T>(phys::Constants<T>::TERRAIN_HEIGHT);
        }
        parameters.terrain.obstacle_count = 0;
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters(DEVICE& device, const rl::environments::BipedalWalkerHardcore<SPEC>& env,
                                                               typename rl::environments::BipedalWalkerHardcore<SPEC>::Parameters& parameters, RNG& rng) {
        using T = typename SPEC::T;
        parameters.geometry = ::bipedal_walker::physics::make_geometry<T>();
        rl::environments::bipedal_walker::TerrainRng<DEVICE, RNG> terrain_rng{rng};
        ::bipedal_walker::physics::generate_terrain_hardcore(parameters.terrain, terrain_rng);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void initial_state(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                   typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters,
                                                   typename rl::environments::BipedalWalker<SPEC>::State& state) {
        using T = typename SPEC::T;
        ::bipedal_walker::physics::reset_walker(parameters.geometry, parameters.terrain, static_cast<T>(0), state);
    }

    // Estado tras reset(): el original aplica una fuerza horizontal uniforme en [-5, 5] al casco.
    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                          typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters,
                                                          typename rl::environments::BipedalWalker<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        const T limit = ::bipedal_walker::physics::Constants<T>::INITIAL_RANDOM;
        const T force_x = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -limit, limit, rng);
        ::bipedal_walker::physics::reset_walker(parameters.geometry, parameters.terrain, force_x, state);
    }

    // Un paso de 1/50 s. La acción son 4 torques en [-1, 1] (cadera y rodilla de cada pierna).
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                      typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters,
                                                      const typename rl::environments::BipedalWalker<SPEC>::State& state,
                                                      const Matrix<ACTION_SPEC>& action,
                                                      typename rl::environments::BipedalWalker<SPEC>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 4);
        using T = typename SPEC::T;
        using PARAMS = typename rl::environments::BipedalWalker<SPEC>::Parameters;
        namespace phys = ::bipedal_walker::physics;

        T a[4];
        for (int k = 0; k < 4; ++k) {
            a[k] = get(action, 0, k);
        }
        phys::JointCommand<T> cmd[phys::JOINT_COUNT];
        phys::joint_commands(a, cmd);

        next_state = state;
        phys::SolverConfig cfg;
        cfg.velocity_iterations = static_cast<int>(PARAMS::VELOCITY_ITERATIONS);
        cfg.position_iterations = static_cast<int>(PARAMS::POSITION_ITERATIONS);
        phys::step_walker(parameters.geometry, parameters.terrain, next_state, cmd, phys::Vec2<T>{0, 0}, cfg);
        return PARAMS::DT;
    }

    // Avance del casco menos penalización por inclinación y por esfuerzo; -100 al caer o salirse
    // por la izquierda.
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                        const typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters,
                                                        const typename rl::environments::BipedalWalker<SPEC>::State& state,
                                                        const Matrix<ACTION_SPEC>& action,
                                                        const typename rl::environments::BipedalWalker<SPEC>::State& next_state, RNG& rng) {
        using T = typename SPEC::T;
        namespace phys = ::bipedal_walker::physics;
        using C = phys::Constants<T>;

        if (next_state.game_over || phys::body_origin(parameters.geometry, next_state.bodies, 0).x < 0) {
            return -100;
        }

        T reward = rl::environments::bipedal_walker::shaping_delta(parameters.geometry, state.bodies, next_state.bodies);
        for (int k = 0; k < 4; ++k) {
            // Coste de esfuerzo, normalizado en el original a ~-50 con su heurística.
            reward -= static_cast<T>(0.00035) * C::MOTORS_TORQUE * phys::clamp(phys::abs(get(action, 0, k)), T(0), T(1));
        }
        return reward;
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                             const typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters,
                                             const typename rl::environments::BipedalWalker<SPEC>::State& state,
                                             const rl::environments::bipedal_walker::Observation<OBS_TYPE_SPEC>&,
                                             Matrix<OBS_SPEC>& observation, RNG& rng) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 24);
        using T = typename SPEC::T;
        namespace phys = ::bipedal_walker::physics;
        using C = phys::Constants<T>;

        const phys::BodyState<T>* b = state.bodies;
        phys::RevoluteJointDef<T> joints[phys::JOINT_COUNT];
        phys::make_joints(joints);
        auto angle = [&](int j) { return b[joints[j].b].a - b[joints[j].a].a - joints[j].reference_angle; };
        auto speed = [&](int j) { return b[joints[j].b].w - b[joints[j].a].w; };

        const T fps = static_cast<T>(C::FPS);
        set(observation, 0, 0, b[0].a);
        set(observation, 0, 1, 2 * b[0].w / fps);
        set(observation, 0, 2, static_cast<T>(0.3) * b[0].v.x * static_cast<T>(C::VIEWPORT_W / C::SCALE) / fps);
        set(observation, 0, 3, static_cast<T>(0.3) * b[0].v.y * static_cast<T>(C::VIEWPORT_H / C::SCALE) / fps);
        set(observation, 0, 4, angle(0));
        set(observation, 0, 5, speed(0) / C::SPEED_HIP);
        set(observation, 0, 6, angle(1) + 1);
        set(observation, 0, 7, speed(1) / C::SPEED_KNEE);
        set(observation, 0, 8, state.leg_contact[0] ? T(1) : T(0));
        set(observation, 0, 9, angle(2));
        set(observation, 0, 10, speed(2) / C::SPEED_HIP);
        set(observation, 0, 11, angle(3) + 1);
        set(observation, 0, 12, speed(3) / C::SPEED_KNEE);
        set(observation, 0, 13, state.leg_contact[1] ? T(1) : T(0));

        T lidar[phys::LIDAR_RAYS];
        phys::lidar_scan(parameters.terrain, phys::body_origin(parameters.geometry, b, 0), lidar);
        for (int i = 0; i < phys::LIDAR_RAYS; ++i) {
            set(observation, 0, 14 + i, lidar[i]);
        }
    }

    // El episodio termina al caer (el casco toca el suelo), al salirse por la izquierda o al
    // llegar al final del terreno. El límite de 1600 pasos es un truncamiento (EPISODE_STEP_LIMIT).
    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool terminated(DEVICE& device, const rl::environments::BipedalWalker<SPEC>& env,
                                                const typename rl::environments::BipedalWalker<SPEC>::Parameters& parameters,
                                                const typename rl::environments::BipedalWalker<SPEC>::State& state, RNG& rng) {
        using T = typename SPEC::T;
        namespace phys = ::bipedal_walker::physics;
        using C = phys::Constants<T>;

        const T x = phys::body_origin(parameters.geometry, state.bodies, 0).x;
        return state.game_over || x < 0 ||
               static_cast<double>(x) > (C::TERRAIN_LENGTH - C::TERRAIN_GRASS) * C::TERRAIN_STEP;
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
