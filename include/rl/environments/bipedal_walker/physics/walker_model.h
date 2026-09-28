// Constantes y geometría del BipedalWalker-v3 de Gymnasium (MIT, Oleg Klimov / Andrea Pierré).
// Las unidades son las del original: SCALE = 30 px por metro, los polígonos están en píxeles.
//
// Los valores se calculan en double y se redondean a T, igual que Python (double) -> Box2D
// (float32), para que los vértices sean idénticos bit a bit cuando T = float.
#pragma once

#include "body_mass.h"

namespace bipedal_walker::physics {

    template <typename T>
    struct Constants {
        static constexpr int FPS = 50;
        static constexpr double SCALE = 30.0;

        static constexpr T MOTORS_TORQUE = 80;
        static constexpr T SPEED_HIP = 4;
        static constexpr T SPEED_KNEE = 6;
        static constexpr T LIDAR_RANGE = static_cast<T>(160.0 / SCALE);
        static constexpr T INITIAL_RANDOM = 5;

        static constexpr int HULL_COUNT = 5;
        static constexpr int HULL_POLY[HULL_COUNT][2] = {{-30, +9}, {+6, +9}, {+34, +1}, {+34, -8}, {-30, -8}};
        static constexpr T LEG_DOWN = static_cast<T>(-8.0 / SCALE);
        static constexpr T LEG_W = static_cast<T>(8.0 / SCALE);
        static constexpr T LEG_H = static_cast<T>(34.0 / SCALE);

        static constexpr T HULL_DENSITY = 5;
        static constexpr T LEG_DENSITY = 1;

        // Terreno. El generador original trabaja en double (floats de Python), así que estas
        // constantes lo son también; solo las alturas finales se redondean a T.
        static constexpr int VIEWPORT_W = 600;
        static constexpr int VIEWPORT_H = 400;
        static constexpr double TERRAIN_STEP = 14.0 / SCALE;
        static constexpr int TERRAIN_LENGTH = 200;  // en pasos
        static constexpr double TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4;
        static constexpr int TERRAIN_GRASS = 10;    // longitud típica de un tramo de hierba
        static constexpr int TERRAIN_STARTPAD = 20;
        static constexpr T FRICTION = static_cast<T>(2.5);          // terreno
        static constexpr T HULL_FRICTION = static_cast<T>(0.1);
        static constexpr T LEG_FRICTION = static_cast<T>(0.2);      // valor por defecto de fixtureDef
    };

    // Formas y masas de los tres tipos de cuerpo. Las dos piernas son idénticas, así que se
    // guarda un único par (muslo, pantorrilla).
    template <typename T>
    struct Geometry {
        Polygon<T, Constants<T>::HULL_COUNT> hull;
        Polygon<T, 4> upper;
        Polygon<T, 4> lower;
        BodyMass<T> hull_mass;
        BodyMass<T> upper_mass;
        BodyMass<T> lower_mass;
    };

    template <typename T>
    Geometry<T> make_geometry() {
        using C = Constants<T>;
        Geometry<T> g;

        Vec2<T> hull_points[C::HULL_COUNT];
        for (int i = 0; i < C::HULL_COUNT; ++i) {
            hull_points[i] = {static_cast<T>(C::HULL_POLY[i][0] / C::SCALE),
                              static_cast<T>(C::HULL_POLY[i][1] / C::SCALE)};
        }
        set_convex_hull(g.hull, hull_points, C::HULL_COUNT);

        // LEG_FD: box(LEG_W / 2, LEG_H / 2); LOWER_FD: box(0.8 * LEG_W / 2, LEG_H / 2).
        // La cuenta se hace en double como en Python antes de redondear a T.
        set_as_box(g.upper, static_cast<T>(8.0 / C::SCALE / 2), static_cast<T>(34.0 / C::SCALE / 2));
        set_as_box(g.lower, static_cast<T>(0.8 * (8.0 / C::SCALE) / 2), static_cast<T>(34.0 / C::SCALE / 2));

        g.hull_mass = compute_body_mass(g.hull, C::HULL_DENSITY);
        g.upper_mass = compute_body_mass(g.upper, C::LEG_DENSITY);
        g.lower_mass = compute_body_mass(g.lower, C::LEG_DENSITY);
        return g;
    }

}
