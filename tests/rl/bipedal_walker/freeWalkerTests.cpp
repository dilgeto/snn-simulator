// Fase 4 / S2: el caminante completo en el aire (juntas, motores, límites, integración)
// contra Box2D con warm-starting, colisión continua y sleeping apagados.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "reference_data.h"
#include "rl/environments/bipedal_walker/physics/world.h"

namespace phys = bipedal_walker::physics;

namespace {

    const char* const BODY[phys::BODY_COUNT] = {"hull", "upper0", "lower0", "upper1", "lower1"};

    // Diferencia máxima entre el estado del port y una fila de la traza de Box2D.
    // Compara posición del origen, ángulo, velocidad del centro de masa y velocidad angular.
    double max_error(const phys::Geometry<float>& g, const phys::BodyState<float>* bodies,
                     const reference::Csv& csv, std::size_t row) {
        double err = 0.0;
        for (int i = 0; i < phys::BODY_COUNT; ++i) {
            std::string n = BODY[i];
            phys::Vec2<float> origin = phys::body_origin(g, bodies, i);
            const double got[6] = {origin.x, origin.y, bodies[i].a, bodies[i].v.x, bodies[i].v.y, bodies[i].w};
            const char* col[6] = {"x", "y", "angle", "vx", "vy", "w"};
            for (int k = 0; k < 6; ++k) {
                err = std::max(err, std::fabs(got[k] - csv.get(row, n + "_" + col[k])));
            }
        }
        return err;
    }

    // Simula la traza `file` con el port y devuelve el error máximo por paso.
    std::vector<double> simulate(const std::string& file, const phys::SolverConfig& cfg) {
        auto csv = reference::read_csv(file);
        auto g = phys::make_geometry<float>();
        phys::BodyState<float> bodies[phys::BODY_COUNT];
        phys::initial_bodies(g, bodies);

        std::vector<double> errors;
        errors.push_back(max_error(g, bodies, csv, 0));  // fila -1: pose inicial

        for (std::size_t r = 1; r < csv.rows.size(); ++r) {
            float action[4];
            for (int k = 0; k < 4; ++k) {
                action[k] = static_cast<float>(csv.get(r, "a" + std::to_string(k)));
            }
            phys::JointCommand<float> cmd[phys::JOINT_COUNT];
            phys::joint_commands(action, cmd);
            phys::Vec2<float> force = r == 1 ? phys::Vec2<float>{4.75f, 0.0f} : phys::Vec2<float>{0.0f, 0.0f};
            phys::step_bodies(g, bodies, cmd, force, cfg);
            errors.push_back(max_error(g, bodies, csv, r));
        }
        return errors;
    }

}

TEST_CASE("BipedalWalker: la pose inicial coincide con Box2D", "[bipedal][free]") {
    for (const char* policy : {"zeros", "sine"}) {
        auto errors = simulate(std::string("free_") + policy + ".csv", phys::SolverConfig{0, 0});
        // Con 0 iteraciones solo se comprueba la fila inicial (antes de simular nada).
        CHECK(errors[0] < 1e-6);
    }
}

TEST_CASE("BipedalWalker: caminante en el aire coincide con Box2D", "[bipedal][free]") {
    for (const char* policy : {"zeros", "sine", "random", "sat_pos", "sat_neg", "chatter"}) {
        auto errors = simulate(std::string("free_") + policy + ".csv", phys::SolverConfig{});
        double worst = 0.0;
        for (double e : errors) worst = std::max(worst, e);
        std::size_t first_bad = 0;  // primera fila que se aleja más de 1e-5
        while (first_bad < errors.size() && errors[first_bad] < 1e-5) ++first_bad;
        INFO(policy << ": error máximo " << worst << ", primera fila con error > 1e-5: " << first_bad
                    << " (de " << errors.size() << ")");
        CHECK(worst < 1e-6);
    }
}
