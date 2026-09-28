// Estudio de fidelidad por ranking: evalúa los controladores de ranking_controllers.csv sobre
// el entorno RLTools (no la capa physics/ directamente) a distintas iteraciones del solver, y
// escribe los retornos a CSV. scripts/bipedal_reference/analyze_ranking.py calcula la correlación
// de Spearman contra los retornos de Box2D (ranking_box2d_seed*.csv, mismos controladores y
// terreno) y decide qué configuración de iteraciones preserva el orden de fitness.
//
// Uso: bipedal_ranking_eval <dir_datos> <lista_semillas separadas por coma>
#include <rl_tools/operations/cpu.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rl/environments/bipedal_walker/operations_generic.h"

namespace rlt = rl_tools;
namespace phys = bipedal_walker::physics;

using T = float;
using TI = std::size_t;
using DEVICE = rlt::devices::DefaultCPU;
using Spec = rlt::rl::environments::bipedal_walker::Specification<T, TI>;
using Env = rlt::rl::environments::BipedalWalker<Spec>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 24, false>>;
using ActionMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>>;

namespace {

    struct Csv {
        std::map<std::string, int> col;
        std::vector<std::vector<double>> rows;
    };

    Csv read_csv(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "no se pudo abrir %s\n", path.c_str());
            std::exit(1);
        }
        Csv csv;
        std::string line;
        std::getline(f, line);
        {
            std::stringstream ss(line);
            std::string c;
            int i = 0;
            while (std::getline(ss, c, ',')) csv.col[c] = i++;
        }
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string c;
            std::vector<double> row;
            while (std::getline(ss, c, ',')) row.push_back(std::stod(c));
            csv.rows.push_back(row);
        }
        return csv;
    }

    struct Controller {
        int id;
        T W[4][24];
        T b[4];

        void act(const T obs[24], T action[4]) const {
            for (int k = 0; k < 4; ++k) {
                T z = b[k];
                for (int i = 0; i < 24; ++i) z += W[k][i] * obs[i];
                action[k] = phys::clamp(std::tanh(z), T(-1), T(1));
            }
        }
    };

    std::vector<Controller> load_controllers(const std::string& path) {
        Csv csv = read_csv(path);
        std::vector<Controller> out;
        for (auto& row : csv.rows) {
            Controller c;
            c.id = static_cast<int>(row[csv.col.at("id")]);
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 24; ++i) {
                    c.W[k][i] = static_cast<T>(row[csv.col.at("w" + std::to_string(k * 24 + i))]);
                }
                c.b[k] = static_cast<T>(row[csv.col.at("b" + std::to_string(k))]);
            }
            out.push_back(c);
        }
        return out;
    }

    phys::Terrain<T> load_terrain(const std::string& path) {
        Csv csv = read_csv(path);
        phys::Terrain<T> t;
        for (auto& row : csv.rows) {
            int i = static_cast<int>(row[csv.col.at("i")]);
            t.y[i] = static_cast<T>(row[csv.col.at("y")]);
        }
        return t;
    }

    // Fuerza inicial que reset() sorteó en Box2D para esta semilla (fase "force" de draws_seed*.csv).
    // Se lee aparte de read_csv() porque esas filas tienen columnas de texto (phase, method).
    T load_force(const std::string& path) {
        std::ifstream f(path);
        std::string line;
        std::getline(f, line);  // encabezado
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string phase, method, a0, a1, value;
            std::getline(ss, phase, ',');
            std::getline(ss, method, ',');
            std::getline(ss, a0, ',');
            std::getline(ss, a1, ',');
            std::getline(ss, value, ',');
            if (phase == "force") return static_cast<T>(std::stod(value));
        }
        fprintf(stderr, "sin fuerza grabada en %s\n", path.c_str());
        std::exit(1);
    }

}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s <dir_datos> <semillas separadas por coma>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    std::vector<int> seeds;
    {
        std::stringstream ss(argv[2]);
        std::string s;
        while (std::getline(ss, s, ',')) seeds.push_back(std::stoi(s));
    }

    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 0);
    Env env;

    auto controllers = load_controllers(dir + "/ranking_controllers.csv");

    // (velocidad, posición): mismo barrido que bench_solver.cpp, salvo los extremos más bajos.
    const int configs[][2] = {{180, 60}, {90, 30}, {60, 20}, {40, 15}, {30, 10}, {20, 8}, {12, 4}, {8, 3}};

    for (int seed : seeds) {
        Env::Parameters params;
        params.geometry = phys::make_geometry<T>();
        params.terrain = load_terrain(dir + "/terrain_seed" + std::to_string(seed) + ".csv");
        T force = load_force(dir + "/draws_seed" + std::to_string(seed) + ".csv");

        for (auto& cfg : configs) {
            // Las iteraciones del solver son parámetros de Env::Parameters (VELOCITY/POSITION_ITERATIONS
            // son constexpr en el tipo); aquí se llama a step_walker directamente para variarlas en
            // tiempo de ejecución sin reinstanciar la plantilla por configuración.
            phys::SolverConfig solver_cfg{cfg[0], cfg[1]};

            std::string out_path = dir + "/ranking_port_seed" + std::to_string(seed) + "_v" +
                                   std::to_string(cfg[0]) + "_p" + std::to_string(cfg[1]) + ".csv";
            std::ofstream out(out_path);
            out << "id,return,steps,terminated\n";
            for (const auto& c : controllers) {
                Env::State state;
                phys::reset_walker(params.geometry, params.terrain, force, state);
                double total = 0.0;
                int steps = 0;
                bool term = false;
                for (; steps < static_cast<int>(Env::EPISODE_STEP_LIMIT); ++steps) {
                    Env::Observation obs_type;
                    ObsMatrix obs;
                    rlt::observe(device, env, params, state, obs_type, obs, rng);
                    T obs_flat[24], action_flat[4];
                    for (int i = 0; i < 24; ++i) obs_flat[i] = rlt::get(obs, 0, i);
                    c.act(obs_flat, action_flat);
                    ActionMatrix action;
                    for (int k = 0; k < 4; ++k) rlt::set(action, 0, k, action_flat[k]);

                    Env::State next = state;
                    phys::JointCommand<T> cmd[phys::JOINT_COUNT];
                    T a4[4] = {action_flat[0], action_flat[1], action_flat[2], action_flat[3]};
                    phys::joint_commands(a4, cmd);
                    phys::step_walker(params.geometry, params.terrain, next, cmd, phys::Vec2<T>{0, 0}, solver_cfg);

                    total += rlt::reward(device, env, params, state, action, next, rng);
                    state = next;
                    if (rlt::terminated(device, env, params, state, rng)) {
                        term = true;
                        ++steps;
                        break;
                    }
                }
                out << c.id << "," << total << "," << steps << "," << (term ? 1 : 0) << "\n";
            }
            printf("seed %d  v%d/p%d -> %s\n", seed, cfg[0], cfg[1], out_path.c_str());
        }
    }
    return 0;
}
