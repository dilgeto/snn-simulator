// Terreno del BipedalWalker-v3, portado de BipedalWalker._generate_terrain (modo normal y
// hardcore).
//
// El terreno es un mapa de alturas: LENGTH puntos equiespaciados cada TERRAIN_STEP en x, unidos
// por aristas (y[], LENGTH * sizeof(T) bytes; x[i] = i * TERRAIN_STEP). En modo hardcore, además,
// hasta MAX_OBSTACLES cuadriláteros estáticos (fosos, tocones, escalones) se superponen al
// terreno; su capacidad se fijó con un barrido Monte Carlo sobre el generador de Python (3000
// semillas: máximo 35 obstáculos, media 29 — MAX_OBSTACLES = 48 deja margen amplio). `Terrain`
// vive en Parameters, compartido de solo lectura entre todos los individuos de una generación,
// así que este margen no pesa en el presupuesto de memoria por individuo.
#pragma once

#include "walker_model.h"

namespace bipedal_walker::physics {

    template <typename T, int LENGTH = Constants<T>::TERRAIN_LENGTH, int MAX_OBSTACLES = 48>
    struct Terrain {
        T y[LENGTH];
        int obstacle_count = 0;
        Polygon<T, 4> obstacles[MAX_OBSTACLES];

        // La coordenada x se calcula en double y se redondea, igual que el original.
        static T x(int i) {
            return static_cast<T>(i * Constants<T>::TERRAIN_STEP);
        }
    };

    // Construye un obstáculo cuadrilátero a partir de 4 esquinas (double, en el orden que arma
    // el generador de Python) y lo añade a `terrain` si hay capacidad. set_convex_hull reordena
    // los vértices igual que b2PolygonShape::Set(), así que el resultado coincide con Box2D.
    template <typename T, int LENGTH, int MAX_OBSTACLES>
    void add_obstacle(Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, const double corners[4][2]) {
        if (terrain.obstacle_count >= MAX_OBSTACLES) {
            return;  // ver la nota de capacidad arriba: no debería ocurrir en la práctica
        }
        Vec2<T> ps[4];
        for (int i = 0; i < 4; ++i) {
            ps[i] = {static_cast<T>(corners[i][0]), static_cast<T>(corners[i][1])};
        }
        set_convex_hull(terrain.obstacles[terrain.obstacle_count++], ps, 4);
    }

    // Genera el terreno consumiendo números aleatorios de `rng`, que debe ofrecer:
    //   double uniform(double low, double high)   // [low, high)
    //   int    integers(int low, int high)        // [low, high)
    // El orden y los argumentos de las llamadas son los del original (np_random), de modo que
    // reproduciendo una secuencia grabada se obtiene el mismo terreno.
    template <typename T, int LENGTH, int MAX_OBSTACLES, typename SOURCE>
    void generate_terrain(Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, SOURCE& rng) {
        using C = Constants<T>;

        double velocity = 0.0;
        double y = C::TERRAIN_HEIGHT;
        int counter = C::TERRAIN_STARTPAD;
        bool oneshot = false;

        for (int i = 0; i < LENGTH; ++i) {
            // En modo normal el estado siempre es GRASS; tras cada tramo hay un paso "oneshot"
            // en que la altura no cambia.
            if (!oneshot) {
                double diff = C::TERRAIN_HEIGHT - y;
                double sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
                velocity = 0.8 * velocity + 0.01 * sign;
                if (i > C::TERRAIN_STARTPAD) {
                    velocity += rng.uniform(-1.0, 1.0) / C::SCALE;
                }
                y += velocity;
            }

            oneshot = false;
            terrain.y[i] = static_cast<T>(y);
            counter -= 1;
            if (counter == 0) {
                counter = rng.integers(C::TERRAIN_GRASS / 2, C::TERRAIN_GRASS);
                oneshot = true;
            }
        }
    }

    // Genera el terreno hardcore: mismo mapa de alturas que el modo normal, pero con estado
    // GRASS/STUMP/STAIRS/PIT. `rng` necesita además:
    //   double random()   // uniform [0, 1)
    // Port literal de BipedalWalker._generate_terrain(hardcore=True); ver el comentario de cada
    // rama para la correspondencia con el original.
    template <typename T, int LENGTH, int MAX_OBSTACLES, typename SOURCE>
    void generate_terrain_hardcore(Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, SOURCE& rng) {
        using C = Constants<T>;
        enum State { GRASS, STUMP, STAIRS, PIT };

        State state = GRASS;
        double velocity = 0.0;
        double y = C::TERRAIN_HEIGHT;
        int counter = C::TERRAIN_STARTPAD;
        bool oneshot = false;

        int stair_steps = 0, stair_width = 0, stair_height = 0;
        double original_y = 0.0;

        for (int i = 0; i < LENGTH; ++i) {
            const double x = i * C::TERRAIN_STEP;

            if (state == GRASS && !oneshot) {
                double diff = C::TERRAIN_HEIGHT - y;
                double sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
                velocity = 0.8 * velocity + 0.01 * sign;
                if (i > C::TERRAIN_STARTPAD) {
                    velocity += rng.uniform(-1.0, 1.0) / C::SCALE;
                }
                y += velocity;

            } else if (state == PIT && oneshot) {
                // Dos paredes verticales, una a cada lado del foso (ancho counter*TERRAIN_STEP);
                // el suelo entre medio se hunde 4*TERRAIN_STEP más abajo (rama "PIT && !oneshot").
                counter = rng.integers(3, 5);
                const double step = C::TERRAIN_STEP;
                const double c1[4][2] = {{x, y}, {x + step, y}, {x + step, y - 4 * step}, {x, y - 4 * step}};
                add_obstacle(terrain, c1);
                const double shift = step * counter;
                const double c2[4][2] = {{c1[0][0] + shift, c1[0][1]}, {c1[1][0] + shift, c1[1][1]},
                                         {c1[2][0] + shift, c1[2][1]}, {c1[3][0] + shift, c1[3][1]}};
                add_obstacle(terrain, c2);
                counter += 2;
                original_y = y;

            } else if (state == PIT && !oneshot) {
                y = original_y;
                if (counter > 1) {
                    y -= 4 * C::TERRAIN_STEP;
                }

            } else if (state == STUMP && oneshot) {
                // counter pasa a ser el ancho del tocón (1 o 2 pasos): tras dibujarlo, el tramo
                // STUMP termina en 1-2 iteraciones (ninguna rama mueve `y` mientras state==STUMP,
                // así que la altura queda en meseta hasta la siguiente transición).
                counter = rng.integers(1, 3);
                const double step = C::TERRAIN_STEP;
                const double c[4][2] = {{x, y}, {x + counter * step, y},
                                        {x + counter * step, y + counter * step}, {x, y + counter * step}};
                add_obstacle(terrain, c);

            } else if (state == STAIRS && oneshot) {
                stair_height = rng.random() > 0.5 ? 1 : -1;
                stair_width = rng.integers(4, 5);   // siempre 4 (integers(4,5) es degenerado)
                stair_steps = rng.integers(3, 5);
                original_y = y;
                for (int s = 0; s < stair_steps; ++s) {
                    const double step = C::TERRAIN_STEP;
                    const double sx0 = x + (s * stair_width) * step;
                    const double sx1 = x + ((1 + s) * stair_width) * step;
                    const double sy0 = y + (s * stair_height) * step;
                    const double sy1 = y + (-1 + s * stair_height) * step;
                    const double c[4][2] = {{sx0, sy0}, {sx1, sy0}, {sx1, sy1}, {sx0, sy1}};
                    add_obstacle(terrain, c);
                }
                counter = stair_steps * stair_width;

            } else if (state == STAIRS && !oneshot) {
                double s = stair_steps * stair_width - counter - stair_height;
                double n = s / stair_width;
                y = original_y + (n * stair_height) * C::TERRAIN_STEP;
            }

            oneshot = false;
            terrain.y[i] = static_cast<T>(y);
            counter -= 1;
            if (counter == 0) {
                counter = rng.integers(C::TERRAIN_GRASS / 2, C::TERRAIN_GRASS);
                if (state == GRASS) {
                    state = static_cast<State>(rng.integers(1, 4));  // STUMP, STAIRS o PIT
                    oneshot = true;
                } else {
                    state = GRASS;
                    oneshot = true;
                }
            }
        }
    }

}
