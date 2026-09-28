// Fase 3: terreno y lidar del BipedalWalker frente a Box2D.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/raycast.h"

namespace phys = bipedal_walker::physics;

namespace {

    // Distancia en ulps entre dos floats (para saber cuánto se acerca el port a Box2D).
    std::int64_t ulps(float a, float b) {
        std::int32_t ia, ib;
        std::memcpy(&ia, &a, sizeof a);
        std::memcpy(&ib, &b, sizeof b);
        if (ia < 0) ia = INT32_MIN - ia;
        if (ib < 0) ib = INT32_MIN - ib;
        return std::llabs(static_cast<std::int64_t>(ia) - ib);
    }

    // Reproduce las llamadas grabadas de np_random de la fase "terrain" y verifica que el
    // generador las pide en el mismo orden y con los mismos argumentos.
    struct ReplayRng {
        const reference::Csv& draws;
        std::size_t next = 0;

        // Salta hasta la siguiente llamada de la fase pedida.
        void advance() {
            while (next < draws.rows.size() && draws.text(next, "phase") != "terrain") {
                ++next;
            }
            REQUIRE(next < draws.rows.size());
        }
        double uniform(double low, double high) {
            advance();
            REQUIRE(draws.text(next, "method") == "uniform");
            REQUIRE(draws.get(next, "arg0") == low);
            REQUIRE(draws.get(next, "arg1") == high);
            return draws.get(next++, "value");
        }
        int integers(int low, int high) {
            advance();
            REQUIRE(draws.text(next, "method") == "integers");
            REQUIRE(draws.get(next, "arg0") == low);
            REQUIRE(draws.get(next, "arg1") == high);
            return static_cast<int>(draws.get(next++, "value"));
        }
        // Todas las llamadas de "terrain" deben haberse consumido exactamente.
        bool exhausted() {
            while (next < draws.rows.size() && draws.text(next, "phase") != "terrain") {
                ++next;
            }
            return next == draws.rows.size();
        }
    };

}

TEST_CASE("BipedalWalker: el generador reproduce el terreno de Gymnasium", "[bipedal][terrain]") {
    for (int seed : {0, 1, 2}) {
        INFO("seed " << seed);
        auto draws = reference::read_csv("draws_seed" + std::to_string(seed) + ".csv");
        auto expected = reference::read_csv("terrain_seed" + std::to_string(seed) + ".csv");

        ReplayRng rng{draws};
        phys::Terrain<float> terrain;
        phys::generate_terrain(terrain, rng);
        CHECK(rng.exhausted());

        std::int64_t max_ulps = 0;
        for (int i = 0; i < phys::Constants<float>::TERRAIN_LENGTH; ++i) {
            float y = static_cast<float>(expected.get(i, "y"));
            float x = static_cast<float>(expected.get(i, "x"));
            CHECK(phys::Terrain<float>::x(i) == x);
            max_ulps = std::max(max_ulps, ulps(terrain.y[i], y));
        }
        INFO("máxima diferencia en ulps: " << max_ulps);
        CHECK(max_ulps == 0);
    }
}

TEST_CASE("BipedalWalker: lidar coincide con la observación de Box2D", "[bipedal][lidar]") {
    // Para cada fila de cada traza: lidar desde la posición del casco (origen del cuerpo)
    // sobre el terreno de la semilla == obs[14..23].
    int rays = 0, misses = 0;
    double max_err = 0.0;
    for (int seed : {0, 1, 2}) {
        auto terrain = reference::load_terrain(seed);
        for (const char* policy : {"zeros", "sine", "random", "heuristic"}) {
            auto trace = reference::read_csv(std::string("trace_") + policy + "_seed" + std::to_string(seed) + ".csv");
            for (std::size_t r = 0; r < trace.rows.size(); ++r) {
                phys::Vec2<float> pos = {static_cast<float>(trace.get(r, "hull_x")),
                                         static_cast<float>(trace.get(r, "hull_y"))};
                float got[phys::LIDAR_RAYS];
                phys::lidar_scan(terrain, pos, got);
                for (int i = 0; i < phys::LIDAR_RAYS; ++i) {
                    float want = static_cast<float>(trace.get(r, "obs" + std::to_string(14 + i)));
                    INFO(policy << " seed " << seed << " fila " << r << " rayo " << i);
                    if (want == 1.0f) {
                        REQUIRE(got[i] == 1.0f);  // sin impacto: debe coincidir exactamente
                        ++misses;
                    }
                    float err = std::fabs(got[i] - want);
                    max_err = std::max(max_err, static_cast<double>(err));
                    REQUIRE(err < 1e-6f);
                    ++rays;
                }
            }
        }
    }
    INFO("rayos comprobados: " << rays << ", sin impacto: " << misses << ", error máximo: " << max_err);
    CHECK(rays > 10000);
    CHECK(max_err < 1e-6);
}
