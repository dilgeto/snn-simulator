// Entorno de RLTools (observación, recompensa, terminación, muestreo inicial) frente a Box2D.
#include <catch2/catch_test_macros.hpp>

#include <rl_tools/operations/cpu.h>

#include <cmath>
#include <string>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/operations_generic.h"

namespace rlt = rl_tools;
namespace phys = bipedal_walker::physics;

namespace {

    using T = float;
    using TI = std::size_t;
    using DEVICE = rlt::devices::DefaultCPU;
    using Spec = rlt::rl::environments::bipedal_walker::Specification<T, TI>;
    using Env = rlt::rl::environments::BipedalWalker<Spec>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 24, false>>;
    using ActionMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>>;

    void action_from_row(const reference::Csv& csv, std::size_t row, ActionMatrix& a) {
        for (int k = 0; k < 4; ++k) {
            rlt::set(a, 0, k, static_cast<T>(csv.get(row, "a" + std::to_string(k))));
        }
    }

    Env::Parameters parameters_for_seed(int seed) {
        Env::Parameters p;
        p.geometry = phys::make_geometry<T>();
        p.terrain = reference::load_terrain(seed);
        return p;
    }

}

TEST_CASE("BipedalWalker RLTools: observe reproduce la observación de Box2D", "[bipedal][env]") {
    DEVICE device;
    RNG rng;
    Env env;
    Env::Observation obs_type;
    double worst = 0.0;
    std::size_t checked = 0;
    for (int seed : {0, 1, 2}) {
        Env::Parameters params = parameters_for_seed(seed);
        for (const char* policy : {"zeros", "sine", "random", "heuristic"}) {
            auto csv = reference::read_csv(std::string("trace_") + policy + "_seed" + std::to_string(seed) + ".csv");
            for (std::size_t r = 0; r < csv.rows.size(); ++r) {
                Env::State state;
                static_cast<phys::WalkerState<T>&>(state) = reference::state_from_row(params.geometry, csv, r);
                ObsMatrix obs;
                rlt::observe(device, env, params, state, obs_type, obs, rng);
                for (int i = 0; i < 24; ++i) {
                    double e = std::fabs(static_cast<double>(rlt::get(obs, 0, i)) - csv.get(r, "obs" + std::to_string(i)));
                    INFO(policy << " seed " << seed << " fila " << r << " obs[" << i << "]");
                    REQUIRE(e < 2e-6);
                    worst = std::max(worst, e);
                }
                ++checked;
            }
        }
    }
    INFO("filas comprobadas: " << checked << ", error máximo: " << worst);
    CHECK(checked > 1000);
}

TEST_CASE("BipedalWalker RLTools: reward y terminated coinciden con Box2D", "[bipedal][env]") {
    DEVICE device;
    RNG rng;
    Env env;
    double worst = 0.0;
    std::size_t terminals = 0;
    for (int seed : {0, 1, 2}) {
        Env::Parameters params = parameters_for_seed(seed);
        for (const char* policy : {"zeros", "sine", "random", "heuristic"}) {
            auto csv = reference::read_csv(std::string("trace_") + policy + "_seed" + std::to_string(seed) + ".csv");
            Env::State prev;
            static_cast<phys::WalkerState<T>&>(prev) = reference::state_from_row(params.geometry, csv, 0);
            for (std::size_t r = 1; r < csv.rows.size(); ++r) {
                Env::State next;
                static_cast<phys::WalkerState<T>&>(next) = reference::state_from_row(params.geometry, csv, r);
                ActionMatrix action;
                action_from_row(csv, r, action);

                double got = rlt::reward(device, env, params, prev, action, next, rng);
                double want = csv.get(r, "reward");
                INFO(policy << " seed " << seed << " fila " << r);
                REQUIRE(std::fabs(got - want) < 1e-4);
                worst = std::max(worst, std::fabs(got - want));

                bool term = rlt::terminated(device, env, params, next, rng);
                REQUIRE(term == (csv.get(r, "terminated") != 0));
                terminals += term;
                prev = next;
            }
        }
    }
    INFO("error máximo de reward: " << worst);
    CHECK(terminals >= 6);  // varias trazas terminan por caída
}

TEST_CASE("BipedalWalker RLTools: step + muestreo inicial funcionan de extremo a extremo", "[bipedal][env]") {
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 42);
    Env env;
    Env::Observation obs_type;

    Env::Parameters params;
    rlt::sample_initial_parameters(device, env, params, rng);

    // Terreno: la meseta inicial (20 pasos) es plana a la altura inicial.
    for (int i = 0; i < 20; ++i) {
        CHECK(params.terrain.y[i] == static_cast<T>(phys::Constants<T>::TERRAIN_HEIGHT));
    }

    Env::State state;
    rlt::sample_initial_state(device, env, params, state, rng);
    ObsMatrix obs;
    rlt::observe(device, env, params, state, obs_type, obs, rng);
    CHECK(rlt::get(obs, 0, 8) == 1.0f);   // ambas pantorrillas apoyadas tras reset
    CHECK(rlt::get(obs, 0, 13) == 1.0f);

    // Sin actuar el caminante cae en ~120 pasos con retorno ~-92 (Gymnasium: -92, 115-124 pasos).
    ActionMatrix action;
    for (int k = 0; k < 4; ++k) rlt::set(action, 0, k, 0.0f);
    double total = 0.0;
    std::size_t steps = 0;
    while (steps < Env::EPISODE_STEP_LIMIT) {
        Env::State next;
        T dt = rlt::step(device, env, params, state, action, next, rng);
        CHECK(dt == static_cast<T>(0.02));
        total += rlt::reward(device, env, params, state, action, next, rng);
        state = next;
        ++steps;
        if (rlt::terminated(device, env, params, state, rng)) break;
    }
    INFO("pasos " << steps << ", retorno " << total);
    CHECK(steps > 80);
    CHECK(steps < 160);
    CHECK(total < -80.0);
    CHECK(total > -100.0);
}

TEST_CASE("BipedalWalker RLTools: el terreno depende solo del RNG", "[bipedal][env]") {
    DEVICE device;
    Env env;
    RNG rng1, rng2, rng3;
    rlt::init(device, rng1, 7);
    rlt::init(device, rng2, 7);
    rlt::init(device, rng3, 8);
    Env::Parameters a, b, c;
    rlt::sample_initial_parameters(device, env, a, rng1);
    rlt::sample_initial_parameters(device, env, b, rng2);
    rlt::sample_initial_parameters(device, env, c, rng3);
    bool same_ab = true, same_ac = true;
    for (int i = 0; i < phys::Constants<T>::TERRAIN_LENGTH; ++i) {
        same_ab = same_ab && a.terrain.y[i] == b.terrain.y[i];
        same_ac = same_ac && a.terrain.y[i] == c.terrain.y[i];
    }
    CHECK(same_ab);
    CHECK_FALSE(same_ac);
}

TEST_CASE("BipedalWalkerHardcore RLTools: corre de punta a punta sin NaN", "[bipedal][env][hardcore]") {
    // No compara contra Box2D bit a bit (eso ya lo hacen hardcoreTerrainTests.cpp y
    // hardcoreLidarTests.cpp); aquí solo se valida que el entorno completo -- terreno con
    // obstáculos, física con collide_polygons y la interfaz de RLTools -- corre de forma sana en
    // muchas semillas: sin NaN/inf, terminando dentro del límite de pasos, con al menos algunos
    // episodios usando obstáculos de verdad.
    using HardcoreEnv = rlt::rl::environments::BipedalWalkerHardcore<Spec>;

    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 123);
    HardcoreEnv env;
    HardcoreEnv::Observation obs_type;

    int episodes_with_obstacles = 0;
    for (int trial = 0; trial < 20; ++trial) {
        HardcoreEnv::Parameters params;
        rlt::sample_initial_parameters(device, env, params, rng);
        if (params.terrain.obstacle_count > 0) ++episodes_with_obstacles;

        HardcoreEnv::State state;
        rlt::sample_initial_state(device, env, params, state, rng);

        std::size_t steps = 0;
        for (; steps < HardcoreEnv::EPISODE_STEP_LIMIT; ++steps) {
            ObsMatrix obs;
            rlt::observe(device, env, params, state, obs_type, obs, rng);
            for (int i = 0; i < 24; ++i) {
                REQUIRE(std::isfinite(rlt::get(obs, 0, i)));
            }

            // Controlador trivial: patrón periódico, ejercita motores y límites (como "sine" en
            // las pruebas de física) sin depender de una política entrenada.
            ActionMatrix action;
            const T phase = static_cast<T>(steps) * static_cast<T>(0.2);
            rlt::set(action, 0, 0, std::sin(phase));
            rlt::set(action, 0, 1, static_cast<T>(0.5) * std::sin(phase + T(1)));
            rlt::set(action, 0, 2, -std::sin(phase));
            rlt::set(action, 0, 3, static_cast<T>(-0.5) * std::sin(phase + T(1)));

            HardcoreEnv::State next;
            rlt::step(device, env, params, state, action, next, rng);
            T r = rlt::reward(device, env, params, state, action, next, rng);
            REQUIRE(std::isfinite(r));
            state = next;
            if (rlt::terminated(device, env, params, state, rng)) break;
        }
        INFO("trial " << trial << ": pasos " << steps << ", obstáculos " << params.terrain.obstacle_count);
        CHECK(steps < HardcoreEnv::EPISODE_STEP_LIMIT);
    }
    CHECK(episodes_with_obstacles > 15);  // el modo normal (terreno plano) no debería colarse
}
