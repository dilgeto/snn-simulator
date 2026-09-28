// Lectura de los CSV de referencia volcados por scripts/bipedal_reference/dump_reference.py.
#pragma once

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rl/environments/bipedal_walker/physics/world.h"

namespace reference {

    inline std::string path(const std::string& file) {
        return std::string(BIPEDAL_TEST_DATA_DIR) + "/" + file;
    }

    // CSV con cabecera: acceso por nombre de columna.
    struct Csv {
        std::map<std::string, std::size_t> column;
        std::vector<std::vector<std::string>> rows;

        double get(std::size_t row, const std::string& name) const {
            return std::stod(rows.at(row).at(column.at(name)));
        }
        std::string text(std::size_t row, const std::string& name) const {
            return rows.at(row).at(column.at(name));
        }
    };

    inline Csv read_csv(const std::string& file) {
        std::ifstream f(path(file));
        REQUIRE(f.is_open());
        Csv csv;
        std::string line;
        std::getline(f, line);
        {
            std::istringstream ss(line);
            std::string name;
            std::size_t i = 0;
            while (std::getline(ss, name, ',')) {
                csv.column[name] = i++;
            }
        }
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<std::string> cells;
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                cells.push_back(cell);
            }
            csv.rows.push_back(cells);
        }
        return csv;
    }

    // Terreno de la semilla (alturas exactas de Box2D, redondeadas a float como hace Box2D).
    inline bipedal_walker::physics::Terrain<float> load_terrain(int seed) {
        auto csv = read_csv("terrain_seed" + std::to_string(seed) + ".csv");
        bipedal_walker::physics::Terrain<float> t;
        REQUIRE(csv.rows.size() == static_cast<std::size_t>(bipedal_walker::physics::Constants<float>::TERRAIN_LENGTH));
        for (std::size_t i = 0; i < csv.rows.size(); ++i) {
            t.y[i] = static_cast<float>(csv.get(i, "y"));
        }
        return t;
    }

    inline const char* const* body_names() {
        static const char* const names[bipedal_walker::physics::BODY_COUNT] = {"hull", "upper0", "lower0", "upper1", "lower1"};
        return names;
    }

    // Estado del port a partir de una fila de la traza (origen y velocidad del centro de masa).
    inline bipedal_walker::physics::WalkerState<float> state_from_row(const bipedal_walker::physics::Geometry<float>& g, const Csv& csv, std::size_t row) {
        bipedal_walker::physics::WalkerState<float> s;
        for (int i = 0; i < bipedal_walker::physics::BODY_COUNT; ++i) {
            const std::string n = body_names()[i];
            float a = static_cast<float>(csv.get(row, n + "_angle"));
            bipedal_walker::physics::Vec2<float> origin = {static_cast<float>(csv.get(row, n + "_x")), static_cast<float>(csv.get(row, n + "_y"))};
            s.bodies[i].a = a;
            s.bodies[i].c = bipedal_walker::physics::mul(bipedal_walker::physics::make_rot(a), bipedal_walker::physics::body_mass(g, i).local_center) + origin;
            s.bodies[i].v = {static_cast<float>(csv.get(row, n + "_vx")), static_cast<float>(csv.get(row, n + "_vy"))};
            s.bodies[i].w = static_cast<float>(csv.get(row, n + "_w"));
        }
        s.game_over = csv.get(row, "game_over") != 0;
        s.leg_contact[0] = csv.get(row, "obs8") != 0;
        s.leg_contact[1] = csv.get(row, "obs13") != 0;
        return s;
    }

    inline double max_error(const bipedal_walker::physics::Geometry<float>& g, const bipedal_walker::physics::WalkerState<float>& s, const Csv& csv, std::size_t row) {
        const char* const cols[6] = {"x", "y", "angle", "vx", "vy", "w"};
        double err = 0.0;
        for (int i = 0; i < bipedal_walker::physics::BODY_COUNT; ++i) {
            bipedal_walker::physics::Vec2<float> origin = bipedal_walker::physics::body_origin(g, s.bodies, i);
            const double got[6] = {origin.x, origin.y, s.bodies[i].a, s.bodies[i].v.x, s.bodies[i].v.y, s.bodies[i].w};
            for (int k = 0; k < 6; ++k) {
                err = std::max(err, std::fabs(got[k] - csv.get(row, std::string(body_names()[i]) + "_" + cols[k])));
            }
        }
        return err;
    }


    // Terreno hardcore de la semilla: alturas de hardcore_terrain_seed*.csv y obstáculos de
    // hardcore_obstacles_seed*.csv (no se reproduce el generador; se leen los datos ya volcados,
    // así esta prueba queda desacoplada de generate_terrain_hardcore).
    inline bipedal_walker::physics::Terrain<float> load_hardcore_terrain(int seed) {
        namespace phys = bipedal_walker::physics;
        auto heights = read_csv("hardcore_terrain_seed" + std::to_string(seed) + ".csv");
        auto obstacles = read_csv("hardcore_obstacles_seed" + std::to_string(seed) + ".csv");

        phys::Terrain<float> t;
        REQUIRE(heights.rows.size() == static_cast<std::size_t>(phys::Constants<float>::TERRAIN_LENGTH));
        for (std::size_t i = 0; i < heights.rows.size(); ++i) {
            t.y[i] = static_cast<float>(heights.get(i, "y"));
        }
        for (std::size_t o = 0; o < obstacles.rows.size(); ++o) {
            int count = static_cast<int>(obstacles.get(o, "count"));
            phys::Vec2<float> ps[4];
            for (int v = 0; v < count; ++v) {
                ps[v] = {static_cast<float>(obstacles.get(o, "v" + std::to_string(v) + "x")),
                        static_cast<float>(obstacles.get(o, "v" + std::to_string(v) + "y"))};
            }
            REQUIRE(t.obstacle_count < static_cast<int>(std::size(t.obstacles)));
            // Los vértices ya vienen en el orden de Box2D (Set()); set_convex_hull los reordena
            // igual (es la misma envolvente convexa), así que reproducirlo aquí es redundante pero
            // inofensivo y evita depender de un segundo camino de construcción de Polygon.
            phys::set_convex_hull(t.obstacles[t.obstacle_count++], ps, count);
        }
        return t;
    }

    // Reproduce las llamadas grabadas de np_random de la fase "terrain" de draws_seed*.csv /
    // hardcore_draws_seed*.csv (uniform, integers y, en modo hardcore, random).
    struct ReplayRng {
        const Csv& draws;
        std::size_t next = 0;

        void advance() {
            while (next < draws.rows.size() && draws.text(next, "phase") != "terrain") ++next;
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
        double random() {
            advance();
            REQUIRE(draws.text(next, "method") == "random");
            return draws.get(next++, "value");
        }
        bool exhausted() {
            while (next < draws.rows.size() && draws.text(next, "phase") != "terrain") ++next;
            return next == draws.rows.size();
        }
    };

}