// Lidar del BipedalWalker: 10 rayos desde el origen del casco contra las aristas del terreno.
// Portado de b2EdgeShape::RayCast (Box2D 2.3.x, licencia zlib, Erin Catto) y de la lógica de
// BipedalWalker.step (recorte de fracción: gana la arista más cercana).
#pragma once

#include "terrain.h"

namespace bipedal_walker::physics {

    // b2EdgeShape::RayCast con transformación identidad (el terreno está en coordenadas de
    // mundo). Devuelve true y la fracción t en [0, max_fraction] si el rayo p1->p2 corta la arista.
    template <typename T>
    bool edge_raycast(Vec2<T> v1, Vec2<T> v2, Vec2<T> p1, Vec2<T> p2, T max_fraction, T& fraction) {
        Vec2<T> d = p2 - p1;
        Vec2<T> e = v2 - v1;
        Vec2<T> normal = {e.y, -e.x};
        normalize(normal);

        T numerator = dot(normal, v1 - p1);
        T denominator = dot(normal, d);
        if (denominator == 0) {
            return false;
        }

        T t = numerator / denominator;
        if (t < 0 || max_fraction < t) {
            return false;
        }

        Vec2<T> q = p1 + t * d;

        Vec2<T> r = v2 - v1;
        T rr = dot(r, r);
        if (rr == 0) {
            return false;
        }

        T s = dot(q - v1, r) / rr;
        if (s < 0 || 1 < s) {
            return false;
        }

        fraction = t;
        return true;
    }

    // b2PolygonShape::RayCast con transformación identidad (el polígono ya está en coordenadas de
    // mundo, como los obstáculos del terreno hardcore). Recorta [0, max_fraction] contra cada
    // semiplano del polígono; lo que sobreviva a todos los recortes es la fracción de entrada.
    template <typename T, int CAPACITY>
    bool polygon_raycast(const Polygon<T, CAPACITY>& poly, Vec2<T> p1, Vec2<T> p2, T max_fraction, T& fraction) {
        Vec2<T> d = p2 - p1;
        T lower = 0, upper = max_fraction;
        int index = -1;

        for (int i = 0; i < poly.count; ++i) {
            T numerator = dot(poly.normals[i], poly.vertices[i] - p1);
            T denominator = dot(poly.normals[i], d);

            if (denominator == 0) {
                if (numerator < 0) {
                    return false;
                }
            } else {
                if (denominator < 0 && numerator < lower * denominator) {
                    lower = numerator / denominator;
                    index = i;
                } else if (denominator > 0 && numerator < upper * denominator) {
                    upper = numerator / denominator;
                }
            }
            if (upper < lower) {
                return false;
            }
        }

        if (index >= 0) {
            fraction = lower;
            return true;
        }
        return false;
    }

    // Fracción del rayo p1->p2 hasta la primera arista del terreno (1 si no toca ninguna) o hasta
    // el primer obstáculo hardcore (fosos, tocones, escalones), lo que esté más cerca. Solo se
    // prueban las aristas cuyo tramo en x puede cruzar el rayo, con un margen de una arista a cada
    // lado para no depender del redondeo del índice; los obstáculos, al ser pocos (hasta
    // MAX_OBSTACLES), se prueban todos.
    template <typename T, int LENGTH, int MAX_OBSTACLES>
    T terrain_raycast(const Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, Vec2<T> p1, Vec2<T> p2) {
        using C = Constants<T>;

        T x_min = p1.x < p2.x ? p1.x : p2.x;
        T x_max = p1.x < p2.x ? p2.x : p1.x;
        int i_first = static_cast<int>(std::floor(static_cast<double>(x_min) / C::TERRAIN_STEP)) - 1;
        int i_last = static_cast<int>(std::floor(static_cast<double>(x_max) / C::TERRAIN_STEP)) + 1;
        if (i_first < 0) i_first = 0;
        if (i_last > LENGTH - 2) i_last = LENGTH - 2;

        T fraction = 1;
        for (int i = i_first; i <= i_last; ++i) {
            Vec2<T> v1 = {Terrain<T, LENGTH, MAX_OBSTACLES>::x(i), terrain.y[i]};
            Vec2<T> v2 = {Terrain<T, LENGTH, MAX_OBSTACLES>::x(i + 1), terrain.y[i + 1]};
            T t;
            if (edge_raycast(v1, v2, p1, p2, fraction, t)) {
                fraction = t;
            }
        }
        for (int i = 0; i < terrain.obstacle_count; ++i) {
            T t;
            if (polygon_raycast(terrain.obstacles[i], p1, p2, fraction, t)) {
                fraction = t;
            }
        }
        return fraction;
    }

    constexpr int LIDAR_RAYS = 10;

    // Rellena `out[0..LIDAR_RAYS)` con las fracciones de los 10 rayos lanzados desde `pos` (origen
    // del casco, no su centro de masa). Los rayos no rotan con el casco: van en coordenadas de
    // mundo, con ángulo 1.5 * i / 10 medido desde la vertical hacia abajo.
    template <typename T, int LENGTH, int MAX_OBSTACLES>
    void lidar_scan(const Terrain<T, LENGTH, MAX_OBSTACLES>& terrain, Vec2<T> pos, T* out) {
        using C = Constants<T>;
        for (int i = 0; i < LIDAR_RAYS; ++i) {
            // El desplazamiento se calcula en double con el rango exacto (160 / SCALE), como en
            // Python, y se redondea a T antes de sumarlo a la posición.
            double angle = 1.5 * i / 10.0;
            double range = 160.0 / C::SCALE;
            Vec2<T> p2 = {pos.x + static_cast<T>(std::sin(angle) * range),
                          pos.y - static_cast<T>(std::cos(angle) * range)};
            out[i] = terrain_raycast(terrain, pos, p2);
        }
    }

}
