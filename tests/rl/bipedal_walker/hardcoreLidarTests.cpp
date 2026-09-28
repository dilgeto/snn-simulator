// Lidar en terreno hardcore (aristas + obstáculos) frente a las lecturas de Box2D.
//
// Usa la posición real del casco de una traza de Box2D (columnas hull_x/hull_y) en vez de
// replayar la física del port, así que la prueba es independiente de cualquier caos de orden de
// contacto: solo valida terrain_raycast/polygon_raycast contra la geometría real.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/raycast.h"

namespace phys = bipedal_walker::physics;

TEST_CASE("BipedalWalkerHardcore: el lidar ve los obstáculos igual que Box2D", "[bipedal][hardcore][lidar]") {
    int rays = 0;
    double max_err = 0.0;
    for (int seed : {0, 1, 2}) {
        auto terrain = reference::load_hardcore_terrain(seed);
        REQUIRE(terrain.obstacle_count > 0);

        for (const char* policy : {"zeros", "sine", "random", "heuristic"}) {
            auto trace = reference::read_csv(std::string("hardcore_trace_") + policy + "_seed" + std::to_string(seed) + ".csv");
            for (std::size_t r = 0; r < trace.rows.size(); ++r) {
                phys::Vec2<float> pos = {static_cast<float>(trace.get(r, "hull_x")), static_cast<float>(trace.get(r, "hull_y"))};
                float got[phys::LIDAR_RAYS];
                phys::lidar_scan(terrain, pos, got);
                for (int i = 0; i < phys::LIDAR_RAYS; ++i) {
                    float want = static_cast<float>(trace.get(r, "obs" + std::to_string(14 + i)));
                    INFO(policy << " seed " << seed << " fila " << r << " rayo " << i);
                    if (want == 1.0f) {
                        REQUIRE(got[i] == 1.0f);  // sin impacto: debe coincidir exactamente
                    } else {
                        double err = std::fabs(static_cast<double>(got[i]) - want);
                        max_err = std::max(max_err, err);
                        REQUIRE(err < 1e-6);
                    }
                    ++rays;
                }
            }
        }
    }
    INFO("rayos comprobados: " << rays << ", error máximo: " << max_err);
    CHECK(rays > 5000);
    CHECK(max_err < 1e-6);
}
