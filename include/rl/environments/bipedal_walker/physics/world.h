// Paso de física del caminante: b2World::Step reducido a lo que este entorno usa.
// Portado de b2Island::Solve (Box2D 2.3.x, licencia zlib, Erin Catto).
//
// Cuerpos (índices fijos): 0 casco, 1 muslo izq., 2 pantorrilla izq., 3 muslo der., 4 pantorrilla der.
// Juntas: 0 cadera izq. (0-1), 1 rodilla izq. (1-2), 2 cadera der. (0-3), 3 rodilla der. (3-4).
//
// Sin memoria dinámica ni estado global. Sin warm-starting, colisión continua ni sleeping
// (ver revolute.h y el README de las pruebas para la medición de su efecto).
#pragma once

#include "contact.h"
#include "revolute.h"
#include "terrain.h"
#include "walker_model.h"

namespace bipedal_walker::physics {

    constexpr int BODY_COUNT = 5;
    constexpr int JOINT_COUNT = 4;

    // Orden en que Box2D resuelve las juntas: el de su recorrido al construir la isla, que no es
    // el de creación. b2World::Solve empieza por el último cuerpo creado (pantorrilla der.) y hace
    // una búsqueda en profundidad; cada cuerpo lista sus juntas de la más nueva a la más antigua.
    // El orden importa porque el solver de posición para en cuanto los errores caen bajo la
    // tolerancia, así que el resultado depende de la secuencia.
    constexpr int JOINT_ORDER[JOINT_COUNT] = {3, 2, 0, 1};

    struct SolverConfig {
        int velocity_iterations = 6 * 30;  // valores del entorno original
        int position_iterations = 2 * 30;
    };

    // Objetivo del motor de una junta: velocidad y torque máximo.
    template <typename T>
    struct JointCommand {
        T speed;
        T max_torque;
    };

    // BipedalWalker.step con control_speed = False: la acción da el signo de la velocidad
    // objetivo y la magnitud del torque máximo.
    template <typename T>
    void joint_commands(const T action[4], JointCommand<T> out[JOINT_COUNT]) {
        using C = Constants<T>;
        const T speeds[JOINT_COUNT] = {C::SPEED_HIP, C::SPEED_KNEE, C::SPEED_HIP, C::SPEED_KNEE};
        for (int k = 0; k < JOINT_COUNT; ++k) {
            T a = action[k];
            T sign = a > 0 ? T(1) : (a < 0 ? T(-1) : T(0));
            out[k].speed = speeds[k] * sign;
            out[k].max_torque = C::MOTORS_TORQUE * clamp(abs(a), T(0), T(1));
        }
    }

    template <typename T>
    void make_joints(RevoluteJointDef<T> joints[JOINT_COUNT]) {
        using C = Constants<T>;
        const T half_leg = C::LEG_H / 2;
        const T hip_lower = static_cast<T>(-0.8), hip_upper = static_cast<T>(1.1);
        const T knee_lower = static_cast<T>(-1.6), knee_upper = static_cast<T>(-0.1);

        // Cadera: ancla en el casco (0, LEG_DOWN) y en el extremo superior del muslo (0, LEG_H / 2).
        // Rodilla: extremo inferior del muslo (0, -LEG_H / 2) y superior de la pantorrilla.
        for (int side = 0; side < 2; ++side) {
            int thigh = 1 + 2 * side;
            int shank = 2 + 2 * side;
            // Las piernas nacen con ángulo -0.05 (izquierda) y +0.05 (derecha); la junta lo toma
            // como referencia. Las rodillas nacen alineadas con el muslo (referencia 0).
            const T hip_reference = static_cast<T>(side == 0 ? -0.05 : 0.05);
            joints[2 * side] = {0, thigh, {0, C::LEG_DOWN}, {0, half_leg}, hip_lower, hip_upper, hip_reference};
            joints[2 * side + 1] = {thigh, shank, {0, -half_leg}, {0, half_leg}, knee_lower, knee_upper, T(0)};
        }
    }

    // Masa del cuerpo i (las dos piernas comparten geometría).
    template <typename T>
    const BodyMass<T>& body_mass(const Geometry<T>& g, int i) {
        if (i == 0) return g.hull_mass;
        return (i % 2 == 1) ? g.upper_mass : g.lower_mass;
    }

    // Pose inicial de BipedalWalker.reset(), antes del primer paso de física. Las posiciones se
    // calculan en double como en Python y se redondean a T.
    template <typename T>
    void initial_bodies(const Geometry<T>& g, BodyState<T> bodies[BODY_COUNT]) {
        using C = Constants<T>;
        const double leg_h = 34.0 / C::SCALE;
        const double leg_down = -8.0 / C::SCALE;
        const double init_x = C::TERRAIN_STEP * C::TERRAIN_STARTPAD / 2;
        const double init_y = C::TERRAIN_HEIGHT + 2 * leg_h;

        struct Placement {
            double y;
            double angle;
        };
        const Placement placement[BODY_COUNT] = {
            {init_y, 0.0},
            {init_y - leg_h / 2 - leg_down, -0.05},
            {init_y - leg_h * 3 / 2 - leg_down, -0.05},
            {init_y - leg_h / 2 - leg_down, 0.05},
            {init_y - leg_h * 3 / 2 - leg_down, 0.05},
        };

        for (int i = 0; i < BODY_COUNT; ++i) {
            T angle = static_cast<T>(placement[i].angle);
            Vec2<T> origin = {static_cast<T>(init_x), static_cast<T>(placement[i].y)};
            // El estado guarda el centro de masa: c = origen + R(angulo) * centro_local.
            bodies[i].c = mul(make_rot(angle), body_mass(g, i).local_center) + origin;
            bodies[i].a = angle;
            bodies[i].v = {0, 0};
            bodies[i].w = 0;
        }
    }

    // Origen del cuerpo (b2Body::GetPosition), que es lo que observan el lidar y la posición del casco.
    template <typename T>
    Vec2<T> body_origin(const Geometry<T>& g, const BodyState<T>* bodies, int i) {
        return bodies[i].c - mul(make_rot(bodies[i].a), body_mass(g, i).local_center);
    }

    // Capacidad de contactos activos por paso. Un pie apoyado da 1-2; el casco solo toca en el
    // paso en que termina el episodio. Si hubiera más, se descartan los últimos.
    constexpr int MAX_CONTACTS = 16;

    // Aristas del terreno que toca una pantorrilla. Una pierna abarca a lo sumo 2-3 aristas
    // (0.47 m cada una); 4 sobra.
    constexpr int MAX_TOUCHING_EDGES = 4;
    struct TouchingEdges {
        int count = 0;
        int edge[MAX_TOUCHING_EDGES] = {0, 0, 0, 0};

        bool contains(int e) const {
            for (int k = 0; k < count; ++k) {
                if (edge[k] == e) return true;
            }
            return false;
        }
    };

    // Estado completo del caminante entre pasos: los 5 cuerpos, los flags que el paso deja para
    // la observación y las aristas que tocaba cada pantorrilla. No hay más estado oculto (sin
    // warm-starting, los impulsos de contacto empiezan en 0 en cada paso).
    template <typename T>
    struct WalkerState {
        BodyState<T> bodies[BODY_COUNT];
        bool game_over;         // el casco tocó el terreno
        bool leg_contact[2];    // ground_contact de la pantorrilla izquierda / derecha
        // Aristas tocadas por cada pantorrilla en el paso anterior. El original mantiene
        // ground_contact con eventos Begin/End de contacto, no con "está tocando ahora": si el pie
        // toca dos aristas y deja de tocar una, el End baja el flag aunque siga apoyado en la otra.
        TouchingEdges touching[2];
    };

    // Orden de los cuerpos al construir la isla de Box2D (DFS desde el último cuerpo creado); los
    // contactos entran a la isla en este orden.
    constexpr int ISLAND_BODY_ORDER[BODY_COUNT] = {4, 3, 0, 1, 2};

    // Detección de contactos con el terreno (b2ContactManager::Collide): un manifold por par
    // (cuerpo, arista) y solo se conservan los que tocan (pointCount > 0). Con posiciones de
    // inicio de paso, igual que Box2D, que evalúa los contactos antes de resolver.
    //
    // Orden de resolución: cuerpos en el orden de la isla y, dentro de cada uno, aristas de mayor a
    // menor índice. Es el orden de Box2D cuando los contactos de un cuerpo se crean de izquierda a
    // derecha (los recorre del más nuevo al más antiguo), y es la mejor regla sin estado: en 30
    // episodios con política aleatoria reproduce Box2D sin diferencias en 13, y en el resto
    // discrepa en ~1e-4 al aparecer dos contactos de un mismo cuerpo creados en otro orden. Copiar
    // el orden exacto exige emular los AABB engordados del broadphase (cuándo se crea cada contacto).
    template <typename T, int LENGTH, int MAX_OBSTACLES>
    int collide_walker(const Geometry<T>& g, const Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, WalkerState<T>& s,
                       ContactConstraint<T> contacts[MAX_CONTACTS]) {
        using C = Constants<T>;
        using S = Settings<T>;
        constexpr T RADIUS = 2 * S::POLYGON_RADIUS;
        constexpr T Y_MARGIN = static_cast<T>(0.1);  // descarte rápido: solo optimización

        s.game_over = false;
        TouchingEdges now[2];
        int n = 0;

        auto collide_body = [&](int bi, const auto& poly) {
            const BodyMass<T>& mass = body_mass(g, bi);
            Rot<T> q = make_rot(s.bodies[bi].a);
            Vec2<T> p = body_position(s.bodies[bi], mass, q);

            // Caja del polígono en coordenadas de mundo.
            T x_min = std::numeric_limits<T>::max(), x_max = -x_min;
            T y_min = x_min, y_max = -x_min;
            for (int k = 0; k < poly.count; ++k) {
                Vec2<T> v = mul(q, poly.vertices[k]) + p;
                x_min = min(x_min, v.x);
                x_max = max(x_max, v.x);
                y_min = min(y_min, v.y);
                y_max = max(y_max, v.y);
            }

            int i_first = static_cast<int>(std::floor(static_cast<double>(x_min - RADIUS) / C::TERRAIN_STEP)) - 1;
            int i_last = static_cast<int>(std::floor(static_cast<double>(x_max + RADIUS) / C::TERRAIN_STEP)) + 1;
            if (i_first < 0) i_first = 0;
            if (i_last > LENGTH - 2) i_last = LENGTH - 2;

            const T friction = std::sqrt(C::FRICTION * (bi == 0 ? C::HULL_FRICTION : C::LEG_FRICTION));

            for (int i = i_last; i >= i_first; --i) {
                Vec2<T> v1 = {Terrain<T, LENGTH, MAX_OBSTACLES>::x(i), terrain.y[i]};
                Vec2<T> v2 = {Terrain<T, LENGTH, MAX_OBSTACLES>::x(i + 1), terrain.y[i + 1]};
                if (y_min > max(v1.y, v2.y) + Y_MARGIN || y_max < min(v1.y, v2.y) - Y_MARGIN) {
                    continue;
                }

                Manifold<T> m;
                collide_edge_polygon(m, v1, v2, poly, q, p);
                if (m.count == 0) {
                    continue;
                }

                if (bi == 0) s.game_over = true;
                if (bi == 2 || bi == 4) {
                    TouchingEdges& t = now[bi == 2 ? 0 : 1];
                    if (t.count < MAX_TOUCHING_EDGES) t.edge[t.count++] = i;
                }

                if (n < MAX_CONTACTS) {
                    ContactConstraint<T>& c = contacts[n++];
                    c.body = bi;
                    c.friction = friction;
                    c.manifold = m;
                }
            }

            // Obstáculos del terreno hardcore (fosos, tocones, escalones): se prueban todos, ya
            // que son pocos (hasta MAX_OBSTACLES) y no hay un índice espacial barato como el de
            // las aristas. En modo normal obstacle_count es 0 y este bucle no cuesta nada.
            for (int oi = terrain.obstacle_count - 1; oi >= 0; --oi) {
                const auto& obstacle = terrain.obstacles[oi];
                Manifold<T> m;
                collide_polygons(m, obstacle, poly, q, p);
                if (m.count == 0) {
                    continue;
                }

                if (bi == 0) s.game_over = true;
                // Un obstáculo cuenta como "arista" para el flag ground_contact, con índice
                // negativo para no chocar con los índices reales de arista (que son >= 0).
                if (bi == 2 || bi == 4) {
                    TouchingEdges& t = now[bi == 2 ? 0 : 1];
                    if (t.count < MAX_TOUCHING_EDGES) t.edge[t.count++] = -1 - oi;
                }

                if (n < MAX_CONTACTS) {
                    ContactConstraint<T>& c = contacts[n++];
                    c.body = bi;
                    c.friction = friction;
                    c.manifold = m;
                }
            }
        };

        for (int k = 0; k < BODY_COUNT; ++k) {
            const int bi = ISLAND_BODY_ORDER[k];
            if (bi == 0) {
                collide_body(bi, g.hull);
            } else if (bi % 2 == 1) {
                collide_body(bi, g.upper);
            } else {
                collide_body(bi, g.lower);
            }
        }

        // ground_contact según los eventos de contacto del original: cada arista que empieza a
        // tocarse (Begin) sube el flag y cada una que deja de tocarse (End) lo baja; con varios
        // eventos en el mismo paso gana el último. Box2D recorre sus contactos del más nuevo al más
        // antiguo, aquí de mayor a menor índice de arista (mismo criterio que arriba).
        for (int leg = 0; leg < 2; ++leg) {
            const TouchingEdges& before = s.touching[leg];
            const TouchingEdges& after = now[leg];

            int events[2 * MAX_TOUCHING_EDGES];
            int m = 0;
            for (int k = 0; k < before.count; ++k) events[m++] = before.edge[k];
            for (int k = 0; k < after.count; ++k) {
                if (!before.contains(after.edge[k])) events[m++] = after.edge[k];
            }
            for (int a = 1; a < m; ++a) {  // orden descendente
                int v = events[a], b = a - 1;
                while (b >= 0 && events[b] < v) { events[b + 1] = events[b]; --b; }
                events[b + 1] = v;
            }

            for (int k = 0; k < m; ++k) {
                const bool was = before.contains(events[k]);
                const bool is = after.contains(events[k]);
                if (is && !was) s.leg_contact[leg] = true;   // BeginContact
                if (was && !is) s.leg_contact[leg] = false;  // EndContact
            }
            s.touching[leg] = after;
        }
        return n;
    }

    // b2Island::Solve: integra velocidades, resuelve restricciones de velocidad (juntas y luego
    // contactos), integra posiciones y resuelve restricciones de posición (contactos y luego
    // juntas). Los contactos deben venir ya en el orden de la isla.
    template <typename T>
    void solve_island(const Geometry<T>& g, BodyState<T> bodies[BODY_COUNT], ContactConstraint<T> contacts[],
                      int contact_count, const JointCommand<T> cmd[JOINT_COUNT], Vec2<T> hull_force,
                      const SolverConfig& cfg) {
        using S = Settings<T>;

        const T h = static_cast<T>(1.0 / Constants<T>::FPS);
        const Vec2<T> gravity = {0, -10};

        RevoluteJointDef<T> joint_defs[JOINT_COUNT];
        make_joints(joint_defs);
        RevoluteJointScratch<T> joints[JOINT_COUNT];

        // Integrar velocidades (sin amortiguación: linearDamping = angularDamping = 0).
        for (int i = 0; i < BODY_COUNT; ++i) {
            Vec2<T> force = i == 0 ? hull_force : Vec2<T>{0, 0};
            bodies[i].v += h * (gravity + body_mass(g, i).inv_mass * force);
        }

        for (int k = 0; k < contact_count; ++k) {
            contact_init_velocity(contacts[k], bodies[contacts[k].body], body_mass(g, contacts[k].body));
        }

        for (int j = 0; j < JOINT_COUNT; ++j) {
            joint_init_velocity(joint_defs[j], joints[j], bodies, body_mass(g, joint_defs[j].a), body_mass(g, joint_defs[j].b));
            joints[j].motor_speed = cmd[j].speed;
            joints[j].max_motor_torque = cmd[j].max_torque;
        }

        for (int it = 0; it < cfg.velocity_iterations; ++it) {
            for (int jo = 0; jo < JOINT_COUNT; ++jo) {
                const int j = JOINT_ORDER[jo];
                joint_solve_velocity(joint_defs[j], joints[j], bodies, body_mass(g, joint_defs[j].a),
                                     body_mass(g, joint_defs[j].b), h);
            }
            for (int k = 0; k < contact_count; ++k) {
                contact_solve_velocity(contacts[k], bodies[contacts[k].body], body_mass(g, contacts[k].body));
            }
        }

        // Integrar posiciones, limitando velocidades enormes.
        for (int i = 0; i < BODY_COUNT; ++i) {
            Vec2<T> v = bodies[i].v;
            T w = bodies[i].w;

            Vec2<T> translation = h * v;
            if (dot(translation, translation) > S::MAX_TRANSLATION_SQUARED) {
                T ratio = S::MAX_TRANSLATION / std::sqrt(translation.x * translation.x + translation.y * translation.y);
                v *= ratio;
            }

            T rotation = h * w;
            if (rotation * rotation > S::MAX_ROTATION_SQUARED) {
                T ratio = S::MAX_ROTATION / abs(rotation);
                w *= ratio;
            }

            bodies[i].c += h * v;
            bodies[i].a += h * w;
            bodies[i].v = v;
            bodies[i].w = w;
        }

        for (int it = 0; it < cfg.position_iterations; ++it) {
            T min_separation = 0;
            for (int k = 0; k < contact_count; ++k) {
                min_separation = contact_solve_position(contacts[k], bodies[contacts[k].body],
                                                        body_mass(g, contacts[k].body), min_separation);
            }
            bool contacts_okay = min_separation >= -3 * S::LINEAR_SLOP;

            bool joints_okay = true;
            for (int jo = 0; jo < JOINT_COUNT; ++jo) {
                const int j = JOINT_ORDER[jo];
                bool okay = joint_solve_position(joint_defs[j], joints[j], bodies, body_mass(g, joint_defs[j].a),
                                                 body_mass(g, joint_defs[j].b));
                joints_okay = joints_okay && okay;
            }
            if (contacts_okay && joints_okay) {
                break;
            }
        }
    }

    // Un paso de tiempo 1/FPS sin terreno (caminante en el aire). `hull_force` actúa solo
    // durante este paso (la que el entorno aplica al casco en reset()).
    template <typename T>
    void step_bodies(const Geometry<T>& g, BodyState<T> bodies[BODY_COUNT], const JointCommand<T> cmd[JOINT_COUNT],
                     Vec2<T> hull_force, const SolverConfig& cfg) {
        solve_island<T>(g, bodies, nullptr, 0, cmd, hull_force, cfg);
    }

    // Un paso de tiempo 1/FPS sobre `terrain`: detecta contactos, actualiza los flags de
    // game_over y contacto de pierna, y resuelve la isla.
    template <typename T, int LENGTH, int MAX_OBSTACLES>
    void step_walker(const Geometry<T>& g, const Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, WalkerState<T>& s,
                     const JointCommand<T> cmd[JOINT_COUNT], Vec2<T> hull_force, const SolverConfig& cfg) {
        ContactConstraint<T> contacts[MAX_CONTACTS];
        const int n = collide_walker(g, terrain, s, contacts);
        solve_island(g, s.bodies, contacts, n, cmd, hull_force, cfg);
    }

}
