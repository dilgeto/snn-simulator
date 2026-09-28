// Benchmark de host: coste por paso del BipedalWalker según las iteraciones del solver, y
// tamaño de las estructuras. No mide tiempos absolutos para la ZCU104 (i7 de escritorio, sin FPU
// comparable a la A53 ni las mismas cachés) — sirve para fijar la forma de la curva
// iteraciones -> coste y elegir un punto de partida a recalibrar en la placa.
//
// Uso: bipedal_bench [pasos_por_config]
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "rl/environments/bipedal_walker/physics/raycast.h"
#include "rl/environments/bipedal_walker/physics/reset.h"

namespace phys = bipedal_walker::physics;
using T = float;

namespace {

    // Controlador heurístico de Gymnasium (BipedalWalkerHeuristics.step_heuristic), portado tal
    // cual para tener una carga de trabajo realista (no cae enseguida, como la política cero).
    struct Heuristic {
        enum State { STAY, PUT_DOWN, PUSH_OFF } state = STAY;
        int moving_leg = 0, supporting_leg = 1;
        static constexpr T SPEED = static_cast<T>(0.29);
        static constexpr T SUPPORT_KNEE_ANGLE = static_cast<T>(0.1);
        T supporting_knee_angle = SUPPORT_KNEE_ANGLE;

        void act(const T s[24], T a[4]) {
            const int moving_base = 4 + 5 * moving_leg;
            const int supporting_base = 4 + 5 * supporting_leg;

            T hip_targ[2] = {0, 0}, knee_targ[2] = {0, 0};
            bool has_hip[2] = {false, false}, has_knee[2] = {false, false};
            T hip_todo[2] = {0, 0}, knee_todo[2] = {0, 0};

            if (state == STAY) {
                hip_targ[moving_leg] = static_cast<T>(1.1);
                has_hip[moving_leg] = true;
                knee_targ[moving_leg] = static_cast<T>(-0.6);
                has_knee[moving_leg] = true;
                supporting_knee_angle += static_cast<T>(0.03);
                if (s[2] > SPEED) supporting_knee_angle += static_cast<T>(0.03);
                supporting_knee_angle = phys::min(supporting_knee_angle, SUPPORT_KNEE_ANGLE);
                knee_targ[supporting_leg] = supporting_knee_angle;
                has_knee[supporting_leg] = true;
                if (s[supporting_base + 0] < static_cast<T>(0.10)) state = PUT_DOWN;
            }
            if (state == PUT_DOWN) {
                hip_targ[moving_leg] = static_cast<T>(0.1);
                has_hip[moving_leg] = true;
                knee_targ[moving_leg] = supporting_knee_angle;
                has_knee[moving_leg] = true;
                knee_targ[supporting_leg] = supporting_knee_angle;
                has_knee[supporting_leg] = true;
                if (s[moving_base + 4] != 0) {
                    state = PUSH_OFF;
                    supporting_knee_angle = phys::min(s[moving_base + 2], SUPPORT_KNEE_ANGLE);
                }
            }
            if (state == PUSH_OFF) {
                knee_targ[moving_leg] = supporting_knee_angle;
                has_knee[moving_leg] = true;
                knee_targ[supporting_leg] = 1;
                has_knee[supporting_leg] = true;
                if (s[supporting_base + 2] > static_cast<T>(0.88) || s[2] > static_cast<T>(1.2) * SPEED) {
                    state = STAY;
                    moving_leg = 1 - moving_leg;
                    supporting_leg = 1 - moving_leg;
                }
            }

            if (has_hip[0]) hip_todo[0] = static_cast<T>(0.9) * (hip_targ[0] - s[4]) - static_cast<T>(0.25) * s[5];
            if (has_hip[1]) hip_todo[1] = static_cast<T>(0.9) * (hip_targ[1] - s[9]) - static_cast<T>(0.25) * s[10];
            if (has_knee[0]) knee_todo[0] = 4 * (knee_targ[0] - s[6]) - static_cast<T>(0.25) * s[7];
            if (has_knee[1]) knee_todo[1] = 4 * (knee_targ[1] - s[11]) - static_cast<T>(0.25) * s[12];

            hip_todo[0] -= static_cast<T>(0.9) * (-s[0]) - static_cast<T>(1.5) * s[1];
            hip_todo[1] -= static_cast<T>(0.9) * (-s[0]) - static_cast<T>(1.5) * s[1];
            knee_todo[0] -= 15 * s[3];
            knee_todo[1] -= 15 * s[3];

            a[0] = phys::clamp(static_cast<T>(0.5) * hip_todo[0], T(-1), T(1));
            a[1] = phys::clamp(static_cast<T>(0.5) * knee_todo[0], T(-1), T(1));
            a[2] = phys::clamp(static_cast<T>(0.5) * hip_todo[1], T(-1), T(1));
            a[3] = phys::clamp(static_cast<T>(0.5) * knee_todo[1], T(-1), T(1));
        }
    };

    void observe(const phys::Geometry<T>& g, const phys::WalkerState<T>& s, const phys::Terrain<T>& terrain, T obs[24]) {
        phys::RevoluteJointDef<T> joints[phys::JOINT_COUNT];
        phys::make_joints(joints);
        auto angle = [&](int j) { return s.bodies[joints[j].b].a - s.bodies[joints[j].a].a - joints[j].reference_angle; };
        auto speed = [&](int j) { return s.bodies[joints[j].b].w - s.bodies[joints[j].a].w; };
        using C = phys::Constants<T>;
        const T fps = static_cast<T>(C::FPS);
        obs[0] = s.bodies[0].a;
        obs[1] = 2 * s.bodies[0].w / fps;
        obs[2] = static_cast<T>(0.3) * s.bodies[0].v.x * static_cast<T>(C::VIEWPORT_W / C::SCALE) / fps;
        obs[3] = static_cast<T>(0.3) * s.bodies[0].v.y * static_cast<T>(C::VIEWPORT_H / C::SCALE) / fps;
        obs[4] = angle(0); obs[5] = speed(0) / C::SPEED_HIP;
        obs[6] = angle(1) + 1; obs[7] = speed(1) / C::SPEED_KNEE;
        obs[8] = s.leg_contact[0] ? T(1) : T(0);
        obs[9] = angle(2); obs[10] = speed(2) / C::SPEED_HIP;
        obs[11] = angle(3) + 1; obs[12] = speed(3) / C::SPEED_KNEE;
        obs[13] = s.leg_contact[1] ? T(1) : T(0);
        phys::lidar_scan(terrain, phys::body_origin(g, s.bodies, 0), obs + 14);
    }

    struct RunResult {
        double steps_per_sec;
        long total_steps;
        double mean_episode_steps;
    };

    // Corre `budget_steps` pasos en total, repartidos en episodios completos con el heurístico
    // sobre terreno aleatorio (std::mt19937, no hace falta reproducir Box2D aquí).
    RunResult run(const phys::SolverConfig& cfg, long budget_steps, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        auto g = phys::make_geometry<T>();
        long total = 0;
        int episodes = 0;

        auto t0 = std::chrono::steady_clock::now();
        while (total < budget_steps) {
            phys::Terrain<T> terrain;
            struct RngAdapter {
                std::mt19937& r;
                double uniform(double lo, double hi) { return lo + (hi - lo) * std::uniform_real_distribution<double>(0, 1)(r); }
                int integers(int lo, int hi) { return lo + static_cast<int>(std::uniform_int_distribution<int>(0, hi - lo - 1)(r)); }
            } adapter{rng};
            phys::generate_terrain(terrain, adapter);

            phys::WalkerState<T> s;
            phys::reset_walker(g, terrain, static_cast<T>(std::uniform_real_distribution<double>(-5, 5)(rng)), s);

            Heuristic h;
            for (int step = 0; step < 1600 && total < budget_steps; ++step) {
                T obs[24], action[4];
                observe(g, s, terrain, obs);
                h.act(obs, action);
                phys::JointCommand<T> cmd[phys::JOINT_COUNT];
                phys::joint_commands(action, cmd);
                phys::step_walker(g, terrain, s, cmd, phys::Vec2<T>{0, 0}, cfg);
                ++total;
                const T x = phys::body_origin(g, s.bodies, 0).x;
                if (s.game_over || x < 0) break;
            }
            ++episodes;
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        return {total / secs, total, static_cast<double>(total) / episodes};
    }

}

// Coste puro de un paso, sin el sesgo de que a menos iteraciones el controlador se caiga antes
// (y por tanto reinicie más veces): siempre el mismo caminante recién reiniciado, mismas
// acciones fijas, terreno plano, `n` pasos consecutivos sin reinicio ni terminación.
double pure_step_cost_us(const phys::SolverConfig& cfg, long n) {
    auto g = phys::make_geometry<T>();
    phys::Terrain<T> terrain;
    for (int i = 0; i < phys::Constants<T>::TERRAIN_LENGTH; ++i) terrain.y[i] = static_cast<T>(phys::Constants<T>::TERRAIN_HEIGHT);
    phys::WalkerState<T> s;
    phys::reset_walker(g, terrain, T(2), s);
    T action[4] = {0.6f, -0.4f, -0.6f, 0.4f};
    phys::JointCommand<T> cmd[phys::JOINT_COUNT];
    phys::joint_commands(action, cmd);

    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < n; ++i) {
        phys::step_walker(g, terrain, s, cmd, phys::Vec2<T>{0, 0}, cfg);
        if (s.game_over) phys::reset_walker(g, terrain, T(2), s);  // no debería pasar en llano
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() * 1e6 / n;
}

int main(int argc, char** argv) {
    const long budget = argc > 1 ? std::atol(argv[1]) : 200000;

    printf("sizeof(WalkerState<float>) = %zu B\n", sizeof(phys::WalkerState<T>));
    printf("sizeof(Geometry<float>)    = %zu B  (compartido entre individuos)\n", sizeof(phys::Geometry<T>));
    printf("sizeof(Terrain<float>)     = %zu B  (compartido entre individuos)\n", sizeof(phys::Terrain<T>));
    printf("sizeof(ContactConstraint<float>) x MAX_CONTACTS = %zu B (temporal, en la pila)\n",
           sizeof(phys::ContactConstraint<T>) * phys::MAX_CONTACTS);
    printf("\n%10s %10s %14s %12s %10s\n", "vel_it", "pos_it", "steps/s", "us/paso", "ep. medio");

    printf("\n%10s %10s %14s\n", "vel_it", "pos_it", "us/paso (puro)");
    const int configs0[][2] = {
        {180, 60}, {90, 30}, {60, 20}, {40, 15}, {30, 10}, {20, 8}, {16, 6}, {12, 4}, {8, 3}, {4, 2}, {2, 1},
    };
    for (const auto& c : configs0) {
        pure_step_cost_us({c[0], c[1]}, 2000);  // calentamiento
        printf("%10d %10d %14.3f\n", c[0], c[1], pure_step_cost_us({c[0], c[1]}, 20000));
    }

    // Escalado entre hilos: cada uno con su propio estado (ninguno compartido salvo Geometry, de
    // solo lectura), como en el uso previsto para neuroevolución con OpenMP.
    {
        const phys::SolverConfig cfg{40, 15};
        double baseline = 0;
        for (int nthreads : {1, 2, 4, 8, 12}) {
            std::vector<std::thread> pool;
            std::vector<long> counts(nthreads, 0);
            auto t0 = std::chrono::steady_clock::now();
            for (int t = 0; t < nthreads; ++t) {
                pool.emplace_back([&, t] {
                    auto r = run(cfg, budget / nthreads, 100 + t);
                    counts[t] = r.total_steps;
                });
            }
            for (auto& th : pool) th.join();
            auto t1 = std::chrono::steady_clock::now();
            long total = 0;
            for (long c : counts) total += c;
            double secs = std::chrono::duration<double>(t1 - t0).count();
            double sps = total / secs;
            if (nthreads == 1) baseline = sps;
            printf("hilos=%2d  total steps/s=%10.0f  (x%.2f vs 1 hilo)\n", nthreads, sps, sps / baseline);
        }
        printf("\n");
    }

    printf("\n== episodios con el controlador heurístico (terreno aleatorio, %ld pasos de presupuesto) ==\n", budget);
    // (velocidad, posición): 180/60 es el original; el resto son candidatos para embebido.
    const int configs[][2] = {
        {180, 60}, {90, 30}, {60, 20}, {40, 15}, {30, 10}, {20, 8}, {16, 6}, {12, 4}, {8, 3}, {4, 2},
    };
    for (const auto& c : configs) {
        const int vi = c[0], pi = c[1];
        phys::SolverConfig cfg{vi, pi};
        // Calentamiento corto antes de medir, para que el primer punto no cargue con el arranque.
        run(cfg, budget / 10, 1);
        auto r = run(cfg, budget, 1);
        printf("%10d %10d %14.0f %12.3f %10.1f\n", vi, pi, r.steps_per_sec, 1e6 / r.steps_per_sec, r.mean_episode_steps);
    }
    return 0;
}
