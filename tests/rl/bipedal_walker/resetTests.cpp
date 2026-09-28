// Estado inicial de un episodio (reset) frente a Box2D.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/reset.h"

namespace phys = bipedal_walker::physics;

namespace {

    const char* const BODY[phys::BODY_COUNT] = {"hull", "upper0", "lower0", "upper1", "lower1"};
    const char* const COL[6] = {"x", "y", "angle", "vx", "vy", "w"};

    double state_error(const phys::Geometry<float>& g, const phys::WalkerState<float>& s, const reference::Csv& csv, std::size_t row) {
        double err = 0.0;
        for (int i = 0; i < phys::BODY_COUNT; ++i) {
            phys::Vec2<float> origin = phys::body_origin(g, s.bodies, i);
            const double got[6] = {origin.x, origin.y, s.bodies[i].a, s.bodies[i].v.x, s.bodies[i].v.y, s.bodies[i].w};
            for (int k = 0; k < 6; ++k) {
                err = std::max(err, std::fabs(got[k] - csv.get(row, std::string(BODY[i]) + "_" + COL[k])));
            }
        }
        return err;
    }

    // Fuerza inicial que sorteó reset() para la semilla (fase "force" de draws_seed*.csv).
    float recorded_force(int seed) {
        auto draws = reference::read_csv("draws_seed" + std::to_string(seed) + ".csv");
        for (std::size_t r = 0; r < draws.rows.size(); ++r) {
            if (draws.text(r, "phase") == "force") return static_cast<float>(draws.get(r, "value"));
        }
        FAIL("sin fuerza grabada");
        return 0.0f;
    }

}

TEST_CASE("BipedalWalker: reset reproduce el barrido de fuerzas de Box2D", "[bipedal][reset]") {
    // El terreno bajo el inicio es plano en todas las semillas, así que cualquiera vale.
    auto terrain = reference::load_terrain(0);
    auto g = phys::make_geometry<float>();
    auto sweep = reference::read_csv("reset_sweep.csv");

    double worst = 0.0;
    for (std::size_t r = 0; r < sweep.rows.size(); ++r) {
        phys::WalkerState<float> s;
        phys::reset_walker(g, terrain, static_cast<float>(sweep.get(r, "force")), s);
        double e = state_error(g, s, sweep, r);
        worst = std::max(worst, e);
        INFO("fila " << r << " fuerza " << sweep.get(r, "force") << " error " << e);
        CHECK(e < 5e-6);
        // Ambas pantorrillas quedan apoyadas: ground_contact = 1 (obs[8] y obs[13]).
        CHECK(s.leg_contact[0] == (sweep.get(r, "obs8") != 0));
        CHECK(s.leg_contact[1] == (sweep.get(r, "obs13") != 0));
        CHECK_FALSE(s.game_over);
    }
    INFO("error máximo del barrido: " << worst);
    CHECK(worst < 5e-6);
}

TEST_CASE("BipedalWalker: episodio completo desde reset coincide con Box2D", "[bipedal][reset]") {
    // reset() con la fuerza grabada y luego los pasos de la traza *_exact (política determinista).
    // La tabla difiere ~5e-7 del estado real y la marcha amplifica de forma caótica cualquier
    // perturbación, así que se exige: error pequeño al inicio, flags iguales, y que el episodio
    // termine (caída) en un paso próximo al de Box2D.
    for (const char* policy : {"zeros", "sine", "heuristic"}) {
        for (int seed : {0, 1, 2}) {
            auto csv = reference::read_csv(std::string("trace_") + policy + "_exact_seed" + std::to_string(seed) + ".csv");
            auto terrain = reference::load_terrain(seed);
            auto g = phys::make_geometry<float>();

            phys::WalkerState<float> s;
            phys::reset_walker(g, terrain, recorded_force(seed), s);
            CHECK(state_error(g, s, csv, 0) < 5e-6);
            CHECK(s.leg_contact[0] == (csv.get(0, "obs8") != 0));
            CHECK(s.leg_contact[1] == (csv.get(0, "obs13") != 0));

            constexpr std::size_t EARLY = 30;
            double worst_early = 0.0;
            bool flags_ok_early = true;
            std::size_t ref_end = csv.rows.size(), port_end = csv.rows.size();
            for (std::size_t r = 1; r < csv.rows.size(); ++r) {
                float action[4];
                for (int k = 0; k < 4; ++k) action[k] = static_cast<float>(csv.get(r, "a" + std::to_string(k)));
                phys::JointCommand<float> cmd[phys::JOINT_COUNT];
                phys::joint_commands(action, cmd);
                phys::step_walker(g, terrain, s, cmd, phys::Vec2<float>{0.0f, 0.0f}, phys::SolverConfig{});
                if (r <= EARLY) {
                    worst_early = std::max(worst_early, state_error(g, s, csv, r));
                    flags_ok_early = flags_ok_early && s.leg_contact[0] == (csv.get(r, "obs8") != 0) &&
                                     s.leg_contact[1] == (csv.get(r, "obs13") != 0);
                }
                if (csv.get(r, "terminated") != 0 && ref_end == csv.rows.size()) ref_end = r;
                if (s.game_over && port_end == csv.rows.size()) port_end = r;
                if (port_end != csv.rows.size() && ref_end != csv.rows.size()) break;
            }
            const long diff = static_cast<long>(port_end) - static_cast<long>(ref_end);
            INFO(policy << " seed " << seed << ": error máx. primeros " << EARLY << " pasos " << worst_early
                        << ", termina en " << port_end << " (Box2D: " << ref_end << ")");
            CHECK(worst_early < 5e-4);
            CHECK(flags_ok_early);
            CHECK(std::labs(diff) <= 5);
        }
    }
}
