// Propiedades de masa de un cuerpo con un único fixture, portadas de b2Body::ResetMassData
// (Box2D 2.3.x, licencia zlib). Copyright (c) 2006-2011 Erin Catto http://www.box2d.org
#pragma once

#include "polygon.h"

namespace bipedal_walker::physics {

    template <typename T>
    struct BodyMass {
        T mass;
        T inv_mass;
        T I;      // inercia respecto al CENTRO DE MASA (la que usa el solver)
        T inv_I;
        Vec2<T> local_center;  // centro de masa en coordenadas del cuerpo
    };

    template <typename T, int CAPACITY>
    BodyMass<T> compute_body_mass(const Polygon<T, CAPACITY>& poly, T density) {
        BodyMass<T> b;
        MassData<T> md = compute_mass(poly, density);

        b.mass = md.mass;
        Vec2<T> local_center = md.mass * md.center;
        b.I = md.I;

        b.inv_mass = T(1) / b.mass;
        local_center *= b.inv_mass;

        // Centrar la inercia en el centro de masa.
        b.I -= b.mass * dot(local_center, local_center);
        b.inv_I = T(1) / b.I;
        b.local_center = local_center;
        return b;
    }

    // b2Body::GetInertia(): inercia respecto al origen del cuerpo.
    template <typename T>
    T inertia_about_origin(const BodyMass<T>& b) {
        return b.I + b.mass * dot(b.local_center, b.local_center);
    }

}
