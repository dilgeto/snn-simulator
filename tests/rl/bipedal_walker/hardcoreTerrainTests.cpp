// Terreno hardcore: mapa de alturas y polígonos obstáculo, frente a Box2D.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/terrain.h"

namespace phys = bipedal_walker::physics;

namespace {

    std::int64_t ulps(float a, float b) {
        std::int32_t ia, ib;
        std::memcpy(&ia, &a, sizeof a);
        std::memcpy(&ib, &b, sizeof b);
        if (ia < 0) ia = INT32_MIN - ia;
        if (ib < 0) ib = INT32_MIN - ib;
        return std::llabs(static_cast<std::int64_t>(ia) - ib);
    }

}

TEST_CASE("BipedalWalkerHardcore: el mapa de alturas reproduce el de Box2D", "[bipedal][hardcore][terrain]") {
    for (int seed : {0, 1, 2, 3, 4}) {
        INFO("seed " << seed);
        auto draws = reference::read_csv("hardcore_draws_seed" + std::to_string(seed) + ".csv");
        auto expected = reference::read_csv("hardcore_terrain_seed" + std::to_string(seed) + ".csv");

        reference::ReplayRng rng{draws};
        phys::Terrain<float> terrain;
        phys::generate_terrain_hardcore(terrain, rng);
        CHECK(rng.exhausted());

        std::int64_t max_ulps = 0;
        for (int i = 0; i < phys::Constants<float>::TERRAIN_LENGTH; ++i) {
            float y = static_cast<float>(expected.get(i, "y"));
            max_ulps = std::max(max_ulps, ulps(terrain.y[i], y));
        }
        INFO("máxima diferencia en ulps: " << max_ulps);
        CHECK(max_ulps == 0);
    }
}

TEST_CASE("BipedalWalkerHardcore: los obstáculos reproducen los de Box2D", "[bipedal][hardcore][terrain]") {
    for (int seed : {0, 1, 2, 3, 4}) {
        INFO("seed " << seed);
        auto draws = reference::read_csv("hardcore_draws_seed" + std::to_string(seed) + ".csv");
        auto expected = reference::read_csv("hardcore_obstacles_seed" + std::to_string(seed) + ".csv");

        reference::ReplayRng rng{draws};
        phys::Terrain<float> terrain;
        phys::generate_terrain_hardcore(terrain, rng);

        REQUIRE(static_cast<std::size_t>(terrain.obstacle_count) == expected.rows.size());
        double max_err = 0.0;
        for (std::size_t o = 0; o < expected.rows.size(); ++o) {
            const auto& poly = terrain.obstacles[o];
            REQUIRE(poly.count == static_cast<int>(expected.get(o, "count")));
            for (int v = 0; v < poly.count; ++v) {
                double ex = expected.get(o, "v" + std::to_string(v) + "x");
                double ey = expected.get(o, "v" + std::to_string(v) + "y");
                max_err = std::max(max_err, std::fabs(static_cast<double>(poly.vertices[v].x) - ex));
                max_err = std::max(max_err, std::fabs(static_cast<double>(poly.vertices[v].y) - ey));
            }
        }
        INFO("error máximo de vértice: " << max_err);
        CHECK(max_err < 1e-6);
    }
}
