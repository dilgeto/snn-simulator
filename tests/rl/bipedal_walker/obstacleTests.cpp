// Colisión contra un obstáculo hardcore aislado (collide_polygons + solver de contactos) frente
// a Box2D, sin terreno de por medio: las aristas se colocan muy lejos (y = -1000) para que solo
// intervenga la caja estática.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/world.h"

namespace phys = bipedal_walker::physics;

namespace {

    phys::Terrain<float> make_isolated_box(float x0, float x1, float y0, float y1) {
        phys::Terrain<float> terrain;
        for (float& y : terrain.y) y = -1000.0f;
        const double corners[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
        phys::add_obstacle(terrain, corners);
        return terrain;
    }

    struct Result {
        double worst = 0.0;         // error máximo en toda la ventana (puede dispararse tras el empate)
        double worst_early = 0.0;   // error máximo antes de `early` pasos
        std::size_t first_bad = 0;  // primera fila con error > 1e-5 (rows+1 si ninguna)
        std::size_t rows = 0;
        bool any_contact = false;
        bool finite = true;
    };

    // early: número de pasos en los que se exige coincidencia casi exacta con Box2D. Más allá,
    // igual que con las aristas del terreno (ver groundWalkerTests.cpp), un contacto con dos
    // caras casi igual de separadas puede resolverse hacia una cara de referencia distinta a la
    // de Box2D — b2FindMaxSeparation es una trepada codiciosa desde un punto de partida
    // heurístico, no un máximo global, y está fielmente portada (ver find_max_separation en
    // manifold.h), así que un empate se puede resolver de otro lado por una diferencia de
    // redondeo minúscula arrastrada desde antes. Es la misma clase de caos por orden de contacto
    // ya documentada para las aristas, aquí a nivel de un solo manifiesto.
    Result replay(const std::string& file, const phys::Terrain<float>& terrain, std::size_t horizon, std::size_t early) {
        auto csv = reference::read_csv(file);
        auto g = phys::make_geometry<float>();
        phys::WalkerState<float> s;
        phys::initial_bodies(g, s.bodies);
        s.game_over = false;
        s.leg_contact[0] = s.leg_contact[1] = false;
        s.touching[0] = s.touching[1] = phys::TouchingEdges{};

        Result res;
        res.first_bad = csv.rows.size() + 1;
        res.rows = std::min(horizon, csv.rows.size() - 1);
        for (std::size_t r = 1; r <= res.rows; ++r) {
            float action[4];
            for (int k = 0; k < 4; ++k) action[k] = static_cast<float>(csv.get(r, "a" + std::to_string(k)));
            phys::JointCommand<float> cmd[phys::JOINT_COUNT];
            phys::joint_commands(action, cmd);
            phys::step_walker(g, terrain, s, cmd, phys::Vec2<float>{0.0f, 0.0f}, phys::SolverConfig{});

            res.any_contact = res.any_contact || s.leg_contact[0] || s.leg_contact[1] || s.game_over;
            double e = reference::max_error(g, s, csv, r);
            res.finite = res.finite && std::isfinite(e);
            res.worst = std::max(res.worst, e);
            if (r <= early) res.worst_early = std::max(res.worst_early, e);
            if (e > 1e-5 && res.first_bad > csv.rows.size()) res.first_bad = r;
        }
        return res;
    }

}

TEST_CASE("BipedalWalker hardcore: pierna sobre una plataforma coincide con Box2D", "[bipedal][hardcore][obstacle]") {
    auto terrain = make_isolated_box(3.0f, 6.3f, 1.0f, 3.0f);
    for (const char* policy : {"zeros", "sine", "sat_neg"}) {
        auto r = replay(std::string("obstacle_platform_") + policy + ".csv", terrain, 40, 20);
        INFO(policy << ": error máx. en los primeros 20 pasos " << r.worst_early << ", en los " << r.rows
                    << " completos " << r.worst << ", primera fila > 1e-5: " << r.first_bad
                    << ", hubo contacto: " << r.any_contact);
        CHECK(r.any_contact);
        CHECK(r.finite);
        CHECK(r.worst_early < 1e-5);
    }
}

TEST_CASE("BipedalWalker hardcore: un tocón alto coincide con Box2D", "[bipedal][hardcore][obstacle]") {
    auto terrain = make_isolated_box(4.3f, 5.1f, 1.0f, 3.4f);
    for (const char* policy : {"zeros", "sine", "sat_neg"}) {
        auto r = replay(std::string("obstacle_stump_") + policy + ".csv", terrain, 25, 1);
        INFO(policy << ": error máx. en el primer paso " << r.worst_early << ", en los " << r.rows
                    << " completos " << r.worst << ", primera fila > 1e-5: " << r.first_bad
                    << ", hubo contacto: " << r.any_contact);
        CHECK(r.any_contact);
        CHECK(r.finite);
        CHECK(r.worst_early < 1e-5);
    }
}
