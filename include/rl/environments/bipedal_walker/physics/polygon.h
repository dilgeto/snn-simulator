// Polígono convexo con capacidad fija, portado de b2PolygonShape (Box2D 2.3.x, licencia zlib).
// Copyright (c) 2006-2010 Erin Catto http://www.box2d.org
//
// Se portan SetAsBox, Set (envolvente convexa por gift wrapping), ComputeCentroid y ComputeMass.
// El orden de vértices que produce Set() importa: la colisión posterior depende de él.
#pragma once

#include "math2d.h"

namespace bipedal_walker::physics {

    template <typename T>
    struct MassData {
        T mass;
        Vec2<T> center;
        T I;  // inercia respecto al origen local del cuerpo (no al centro de masa)
    };

    template <typename T, int CAPACITY>
    struct Polygon {
        int count = 0;
        Vec2<T> vertices[CAPACITY];
        Vec2<T> normals[CAPACITY];
        Vec2<T> centroid = {0, 0};
    };

    // b2PolygonShape::SetAsBox(hx, hy): caja centrada en el origen local.
    template <typename T, int CAPACITY>
    void set_as_box(Polygon<T, CAPACITY>& poly, T hx, T hy) {
        static_assert(CAPACITY >= 4);
        poly.count = 4;
        poly.vertices[0] = {-hx, -hy};
        poly.vertices[1] = { hx, -hy};
        poly.vertices[2] = { hx,  hy};
        poly.vertices[3] = {-hx,  hy};
        poly.normals[0] = { 0, -1};
        poly.normals[1] = { 1,  0};
        poly.normals[2] = { 0,  1};
        poly.normals[3] = {-1,  0};
        poly.centroid = {0, 0};
    }

    template <typename T>
    Vec2<T> compute_centroid(const Vec2<T>* vs, int count) {
        Vec2<T> c = {0, 0};
        T area = 0;
        const Vec2<T> p_ref = {0, 0};
        const T inv3 = T(1) / T(3);

        for (int i = 0; i < count; ++i) {
            Vec2<T> p1 = p_ref;
            Vec2<T> p2 = vs[i];
            Vec2<T> p3 = i + 1 < count ? vs[i + 1] : vs[0];

            Vec2<T> e1 = p2 - p1;
            Vec2<T> e2 = p3 - p1;

            T D = cross(e1, e2);
            T triangle_area = T(0.5) * D;
            area += triangle_area;

            c += (triangle_area * inv3) * (p1 + p2 + p3);
        }

        c *= T(1) / area;
        return c;
    }

    // b2PolygonShape::Set(vertices, count): envolvente convexa antihoraria que empieza en el
    // punto más a la derecha (a igual x, el más bajo). `count` debe estar entre 3 y CAPACITY.
    template <typename T, int CAPACITY>
    void set_convex_hull(Polygon<T, CAPACITY>& poly, const Vec2<T>* ps, int n) {
        int i0 = 0;
        T x0 = ps[0].x;
        for (int i = 1; i < n; ++i) {
            T x = ps[i].x;
            if (x > x0 || (x == x0 && ps[i].y < ps[i0].y)) {
                i0 = i;
                x0 = x;
            }
        }

        int hull[CAPACITY];
        int m = 0;
        int ih = i0;

        for (;;) {
            hull[m] = ih;

            int ie = 0;
            for (int j = 1; j < n; ++j) {
                if (ie == ih) {
                    ie = j;
                    continue;
                }

                Vec2<T> r = ps[ie] - ps[hull[m]];
                Vec2<T> v = ps[j] - ps[hull[m]];
                T c = cross(r, v);
                if (c < 0) {
                    ie = j;
                }

                // Colinealidad: quedarse con el punto más lejano.
                if (c == 0 && length_squared(v) > length_squared(r)) {
                    ie = j;
                }
            }

            ++m;
            ih = ie;

            if (ie == i0) {
                break;
            }
        }

        poly.count = m;
        for (int i = 0; i < m; ++i) {
            poly.vertices[i] = ps[hull[i]];
        }
        for (int i = 0; i < m; ++i) {
            int i2 = i + 1 < m ? i + 1 : 0;
            Vec2<T> edge = poly.vertices[i2] - poly.vertices[i];
            poly.normals[i] = cross(edge, T(1));
            normalize(poly.normals[i]);
        }
        poly.centroid = compute_centroid(poly.vertices, m);
    }

    // b2PolygonShape::ComputeMass(density).
    template <typename T, int CAPACITY>
    MassData<T> compute_mass(const Polygon<T, CAPACITY>& poly, T density) {
        Vec2<T> center = {0, 0};
        T area = 0;
        T I = 0;

        // Punto de referencia dentro del polígono (mejor condicionamiento numérico).
        Vec2<T> s = {0, 0};
        for (int i = 0; i < poly.count; ++i) {
            s += poly.vertices[i];
        }
        s *= T(1) / T(poly.count);

        const T k_inv3 = T(1) / T(3);

        for (int i = 0; i < poly.count; ++i) {
            Vec2<T> e1 = poly.vertices[i] - s;
            Vec2<T> e2 = i + 1 < poly.count ? poly.vertices[i + 1] - s : poly.vertices[0] - s;

            T D = cross(e1, e2);
            T triangle_area = T(0.5) * D;
            area += triangle_area;

            center += (triangle_area * k_inv3) * (e1 + e2);

            T ex1 = e1.x, ey1 = e1.y;
            T ex2 = e2.x, ey2 = e2.y;

            T intx2 = ex1 * ex1 + ex2 * ex1 + ex2 * ex2;
            T inty2 = ey1 * ey1 + ey2 * ey1 + ey2 * ey2;

            I += (T(0.25) * k_inv3 * D) * (intx2 + inty2);
        }

        MassData<T> md;
        md.mass = density * area;

        center *= T(1) / area;
        md.center = center + s;

        md.I = density * I;
        // Trasladar al centro de masa y luego al origen del cuerpo.
        md.I += md.mass * (dot(md.center, md.center) - dot(center, center));
        return md;
    }

}
