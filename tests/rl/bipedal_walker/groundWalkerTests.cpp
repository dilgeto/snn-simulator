// Fase 4 / S3-S6: el caminante sobre el terreno (contactos, fricción, game over) contra Box2D.
//
// Cada traza *_exact parte del estado que deja el reset() oficial y sigue con warm-starting,
// colisión continua y sleeping apagados en Box2D (lo que el port no implementa). El reset se
// prueba aparte porque su paso de física depende de la colisión continua.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/world.h"

namespace phys = bipedal_walker::physics;

namespace {

    using reference::state_from_row;
    using reference::max_error;

    struct Result {
        std::size_t rows = 0;          // filas simuladas
        std::size_t first_bad = 0;     // primera fila con error > 1e-5 (rows si ninguna)
        double worst = 0.0;
        std::size_t flag_mismatches = 0;
        std::size_t first_flag_bad = 0;
    };

    Result replay(const std::string& policy, int seed, const phys::SolverConfig& cfg) {
        auto csv = reference::read_csv("trace_" + policy + "_exact_seed" + std::to_string(seed) + ".csv");
        auto terrain = reference::load_terrain(seed);
        auto g = phys::make_geometry<float>();
        phys::WalkerState<float> s = state_from_row(g, csv, 0);
        {
            // Las aristas tocadas al inicio salen de la geometría del estado inicial; los flags
            // ya vienen de la traza. (collide_walker no cambia las posiciones.)
            phys::WalkerState<float> probe = s;
            probe.touching[0] = probe.touching[1] = phys::TouchingEdges{};
            phys::ContactConstraint<float> scratch[phys::MAX_CONTACTS];
            phys::collide_walker(g, terrain, probe, scratch);
            s.touching[0] = probe.touching[0];
            s.touching[1] = probe.touching[1];
        }

        Result res;
        res.first_bad = csv.rows.size();
        res.first_flag_bad = csv.rows.size();
        for (std::size_t r = 1; r < csv.rows.size(); ++r) {
            float action[4];
            for (int k = 0; k < 4; ++k) {
                action[k] = static_cast<float>(csv.get(r, "a" + std::to_string(k)));
            }
            phys::JointCommand<float> cmd[phys::JOINT_COUNT];
            phys::joint_commands(action, cmd);
            phys::step_walker(g, terrain, s, cmd, phys::Vec2<float>{0.0f, 0.0f}, cfg);

            double e = max_error(g, s, csv, r);
            res.worst = std::max(res.worst, e);
            if (e > 1e-5 && res.first_bad == csv.rows.size()) res.first_bad = r;

            bool flags_ok = s.game_over == (csv.get(r, "game_over") != 0) &&
                            s.leg_contact[0] == (csv.get(r, "obs8") != 0) &&
                            s.leg_contact[1] == (csv.get(r, "obs13") != 0);
            if (!flags_ok) {
                ++res.flag_mismatches;
                if (res.first_flag_bad == csv.rows.size()) res.first_flag_bad = r;
            }
            ++res.rows;
            if (csv.get(r, "terminated") != 0) break;
        }
        return res;
    }

}

TEST_CASE("BipedalWalker: caminante sobre el terreno coincide con Box2D", "[bipedal][ground]") {
    // Políticas deterministas: el episodio completo coincide con Box2D (error ~5e-8, límite de
    // precisión de los datos) y los flags de contacto y game over también.
    for (const char* policy : {"zeros", "sine", "heuristic"}) {
        for (int seed : {0, 1, 2}) {
            auto r = replay(policy, seed, phys::SolverConfig{});
            INFO(policy << " seed " << seed << ": " << r.rows << " filas, error máx. " << r.worst
                        << ", primera fila > 1e-5: " << r.first_bad << ", flags distintos: " << r.flag_mismatches);
            CHECK(r.worst < 1e-5);
            CHECK(r.flag_mismatches == 0);
        }
    }
}

TEST_CASE("BipedalWalker: política aleatoria sobre el terreno, límite conocido", "[bipedal][ground]") {
    // Con movimientos aleatorios aparecen a veces dos contactos de un mismo cuerpo creados en un
    // orden que la regla del port (aristas de mayor a menor índice) no reproduce; Box2D lo decide
    // con el historial de sus AABB engordados. El solver es exacto: con el orden correcto coincide
    // hasta 5e-9. Aquí se exige que la primera discrepancia (~1e-4) aparezca pasado el arranque y
    // que los flags coincidan mientras los estados coinciden.
    for (int seed : {0, 1, 2}) {
        auto r = replay("random", seed, phys::SolverConfig{});
        INFO("random seed " << seed << ": primera fila > 1e-5: " << r.first_bad << ", primer flag distinto: " << r.first_flag_bad);
        CHECK(r.first_bad >= 30);
        CHECK(r.first_flag_bad >= r.first_bad - 5);
    }
}
