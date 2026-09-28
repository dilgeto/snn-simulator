// Fase 2: geometría y masas del BipedalWalker frente a Box2D (datos de constants.txt).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rl/environments/bipedal_walker/physics/walker_model.h"

namespace phys = bipedal_walker::physics;
using Catch::Approx;

namespace {

    // Una línea de constants.txt ya separada en palabras.
    using Line = std::vector<std::string>;

    std::vector<Line> read_constants() {
        std::ifstream f(std::string(BIPEDAL_TEST_DATA_DIR) + "/constants.txt");
        REQUIRE(f.is_open());
        std::vector<Line> lines;
        std::string text;
        while (std::getline(f, text)) {
            std::istringstream ss(text);
            Line words;
            std::string w;
            while (ss >> w) {
                words.push_back(w);
            }
            lines.push_back(words);
        }
        return lines;
    }

    // Busca la línea "<tag> <name> ..." (p. ej. "body hull ...").
    const Line& find(const std::vector<Line>& lines, const std::string& tag, const std::string& name) {
        for (const Line& l : lines) {
            if (l.size() >= 2 && l[0] == tag && l[1] == name) {
                return l;
            }
        }
        FAIL("no está en constants.txt: " << tag << " " << name);
        static const Line empty;
        return empty;
    }

    // Valor(es) que siguen a la palabra clave `key` en la línea.
    std::vector<double> after(const Line& l, const std::string& key, std::size_t n) {
        for (std::size_t i = 0; i < l.size(); ++i) {
            if (l[i] == key) {
                std::vector<double> v;
                for (std::size_t k = 1; k <= n; ++k) {
                    v.push_back(std::stod(l[i + k]));
                }
                return v;
            }
        }
        FAIL("falta la clave " << key);
        return {};
    }

    // float32 exacto, o casi: 1e-6 relativo con margen absoluto 1e-7 para ceros.
    Approx near(float expected) {
        return Approx(static_cast<double>(expected)).epsilon(1e-6).margin(1e-7);
    }

    template <int N>
    void check_shape(const std::vector<Line>& lines, const std::string& name, const phys::Polygon<float, N>& poly) {
        const Line& l = find(lines, "shape", name);
        int count = static_cast<int>(after(l, "count", 1)[0]);
        REQUIRE(poly.count == count);
        std::vector<double> verts = after(l, "vertices", 2 * count);
        for (int i = 0; i < count; ++i) {
            INFO(name << " vértice " << i);
            CHECK(static_cast<double>(poly.vertices[i].x) == near(static_cast<float>(verts[2 * i])));
            CHECK(static_cast<double>(poly.vertices[i].y) == near(static_cast<float>(verts[2 * i + 1])));
        }
    }

    // Las normales van después de "normals" y `after` las lee en orden.
    template <int N>
    void check_normals(const std::vector<Line>& lines, const std::string& name, const phys::Polygon<float, N>& poly) {
        const Line& l = find(lines, "shape", name);
        std::vector<double> nrm = after(l, "normals", 2 * poly.count);
        for (int i = 0; i < poly.count; ++i) {
            INFO(name << " normal " << i);
            CHECK(static_cast<double>(poly.normals[i].x) == near(static_cast<float>(nrm[2 * i])));
            CHECK(static_cast<double>(poly.normals[i].y) == near(static_cast<float>(nrm[2 * i + 1])));
        }
    }

    void check_mass(const std::vector<Line>& lines, const std::string& name, const phys::BodyMass<float>& m) {
        const Line& l = find(lines, "body", name);
        INFO(name);
        CHECK(static_cast<double>(m.mass) == near(static_cast<float>(after(l, "mass", 1)[0])));
        CHECK(static_cast<double>(phys::inertia_about_origin(m)) == near(static_cast<float>(after(l, "inertia", 1)[0])));
        CHECK(static_cast<double>(m.I) == near(static_cast<float>(after(l, "inertia_com", 1)[0])));
        std::vector<double> lc = after(l, "local_center", 2);
        CHECK(static_cast<double>(m.local_center.x) == near(static_cast<float>(lc[0])));
        CHECK(static_cast<double>(m.local_center.y) == near(static_cast<float>(lc[1])));
    }

}

TEST_CASE("BipedalWalker: formas coinciden con Box2D", "[bipedal][geometry]") {
    auto lines = read_constants();
    auto g = phys::make_geometry<float>();

    // El orden de vértices del casco lo fija Set() (gift wrapping): parte del punto más a la derecha.
    check_shape(lines, "hull", g.hull);
    check_normals(lines, "hull", g.hull);
    check_shape(lines, "upper0", g.upper);
    check_normals(lines, "upper0", g.upper);
    check_shape(lines, "lower0", g.lower);
    check_normals(lines, "lower0", g.lower);
    // Las dos piernas comparten geometría.
    check_shape(lines, "upper1", g.upper);
    check_shape(lines, "lower1", g.lower);
}

TEST_CASE("BipedalWalker: masas e inercias coinciden con Box2D", "[bipedal][mass]") {
    auto lines = read_constants();
    auto g = phys::make_geometry<float>();

    check_mass(lines, "hull", g.hull_mass);
    check_mass(lines, "upper0", g.upper_mass);
    check_mass(lines, "lower0", g.lower_mass);
    check_mass(lines, "upper1", g.upper_mass);
    check_mass(lines, "lower1", g.lower_mass);

    // El casco es asimétrico: su centro de masa no coincide con el origen del cuerpo.
    CHECK(g.hull_mass.local_center.x != 0.0f);
    CHECK(g.hull_mass.local_center.y != 0.0f);
}

TEST_CASE("BipedalWalker: constantes del módulo", "[bipedal][constants]") {
    auto lines = read_constants();
    using C = phys::Constants<float>;

    auto scalar = [&](const std::string& key) {
        for (const Line& l : lines) {
            if (l.size() == 2 && l[0] == key) {
                return static_cast<float>(std::stod(l[1]));
            }
        }
        FAIL("falta " << key);
        return 0.0f;
    };

    CHECK(static_cast<float>(C::FPS) == scalar("FPS"));
    CHECK(C::MOTORS_TORQUE == scalar("MOTORS_TORQUE"));
    CHECK(C::SPEED_HIP == scalar("SPEED_HIP"));
    CHECK(C::SPEED_KNEE == scalar("SPEED_KNEE"));
    CHECK(C::LIDAR_RANGE == scalar("LIDAR_RANGE"));
    CHECK(C::LEG_DOWN == scalar("LEG_DOWN"));
    CHECK(C::LEG_W == scalar("LEG_W"));
    CHECK(C::LEG_H == scalar("LEG_H"));
}
