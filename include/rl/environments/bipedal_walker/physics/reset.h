// Estado inicial de un episodio: lo que deja BipedalWalker.reset() del original.
//
// reset() construye el caminante 0.27 m sobre el suelo, le aplica una fuerza horizontal aleatoria
// y ejecuta un paso de física con acción 0. Ese paso depende de la colisión continua (TOI): la
// corrección de las juntas hundiría las pantorrillas 0.2 m en el suelo y TOI las detiene en el
// contacto. En vez de portar TOI, se usa el estado resultante, precalculado con Box2D en función
// de la fuerza (reset_table.h): el suelo bajo el inicio es siempre plano, así que es lo único que
// importa (no depende de la semilla ni del terreno más allá de la meseta inicial).
#pragma once

#include "reset_table.h"
#include "world.h"

namespace bipedal_walker::physics {

    // Estado tras reset() con fuerza inicial `force_x` en el casco, sobre `terrain`. La fuerza
    // debe estar en [-INITIAL_RANDOM, INITIAL_RANDOM] (el original la sortea uniforme ahí).
    template <typename T, int LENGTH, int MAX_OBSTACLES>
    void reset_walker(const Geometry<T>& g, const Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, T force_x, WalkerState<T>& s) {
        using R = ResetTable<T>;
        const T* slope = force_x < 0 ? R::SLOPE_NEG : R::SLOPE_POS;

        for (int i = 0; i < BODY_COUNT; ++i) {
            T v[6];
            for (int k = 0; k < 6; ++k) {
                v[k] = R::S0[6 * i + k] + force_x * slope[6 * i + k];
            }
            const Vec2<T> origin = {v[0], v[1]};
            s.bodies[i].a = v[2];
            // El estado guarda el centro de masa: c = origen + R(angulo) * centro_local.
            s.bodies[i].c = mul(make_rot(v[2]), body_mass(g, i).local_center) + origin;
            s.bodies[i].v = {v[3], v[4]};
            s.bodies[i].w = v[5];
        }

        // Los contactos y flags iniciales salen de la geometría: las pantorrillas ya apoyadas
        // generan los eventos Begin que dejan ground_contact = 1 (como en el original).
        s.game_over = false;
        s.leg_contact[0] = s.leg_contact[1] = false;
        s.touching[0] = s.touching[1] = TouchingEdges{};
        ContactConstraint<T> scratch[MAX_CONTACTS];
        collide_walker(g, terrain, s, scratch);
    }

}
