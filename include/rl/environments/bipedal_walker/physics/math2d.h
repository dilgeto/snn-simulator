// Matemática 2D mínima, portada de Box2D 2.3.x (b2Math.h, licencia zlib).
// Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
//
// Solo lo que el BipedalWalker necesita. Sin asignación dinámica ni estado global: todo es
// plantilla sobre T (float en el objetivo embebido) y se comporta igual que b2Vec2 en float32.
#pragma once

#include <cmath>
#include <limits>

namespace bipedal_walker::physics {

    template <typename T>
    struct Vec2 {
        T x;
        T y;
    };

    template <typename T>
    constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }

    template <typename T>
    constexpr Vec2<T> operator+(Vec2<T> a, Vec2<T> b) { return {a.x + b.x, a.y + b.y}; }
    template <typename T>
    constexpr Vec2<T> operator-(Vec2<T> a, Vec2<T> b) { return {a.x - b.x, a.y - b.y}; }
    template <typename T>
    constexpr Vec2<T> operator-(Vec2<T> a) { return {-a.x, -a.y}; }
    template <typename T>
    constexpr Vec2<T> operator*(T s, Vec2<T> a) { return {s * a.x, s * a.y}; }
    template <typename T>
    constexpr Vec2<T>& operator+=(Vec2<T>& a, Vec2<T> b) { a.x += b.x; a.y += b.y; return a; }
    template <typename T>
    constexpr Vec2<T>& operator-=(Vec2<T>& a, Vec2<T> b) { a.x -= b.x; a.y -= b.y; return a; }
    template <typename T>
    constexpr Vec2<T>& operator*=(Vec2<T>& a, T s) { a.x *= s; a.y *= s; return a; }

    template <typename T>
    constexpr T dot(Vec2<T> a, Vec2<T> b) { return a.x * b.x + a.y * b.y; }
    template <typename T>
    constexpr T cross(Vec2<T> a, Vec2<T> b) { return a.x * b.y - a.y * b.x; }
    // Perpendicular a la derecha: cross(a, 1).
    template <typename T>
    constexpr Vec2<T> cross(Vec2<T> a, T s) { return {s * a.y, -s * a.x}; }
    // Perpendicular a la izquierda: cross(1, a). Es la velocidad de un punto a distancia a.
    template <typename T>
    constexpr Vec2<T> cross(T s, Vec2<T> a) { return {-s * a.y, s * a.x}; }

    template <typename T>
    constexpr T length_squared(Vec2<T> a) { return a.x * a.x + a.y * a.y; }

    // Igual que b2Vec2::Normalize: devuelve la longitud original y deja (0,0) si es casi nula.
    template <typename T>
    T normalize(Vec2<T>& a) {
        T length = std::sqrt(a.x * a.x + a.y * a.y);
        if (length < epsilon<T>()) {
            return 0;
        }
        T inv_length = 1 / length;
        a.x *= inv_length;
        a.y *= inv_length;
        return length;
    }

    // b2Min / b2Max / b2Clamp (mismos criterios de comparación que Box2D, importan con NaN y -0).
    template <typename T>
    constexpr T min(T a, T b) { return a < b ? a : b; }
    template <typename T>
    constexpr T max(T a, T b) { return a > b ? a : b; }
    template <typename T>
    constexpr T clamp(T a, T low, T high) { return max(low, min(a, high)); }
    template <typename T>
    constexpr T abs(T a) { return a > 0 ? a : -a; }

    // b2Rot: rotación como (seno, coseno). Box2D usa sinf/cosf, aquí std::sin/cos sobre T.
    template <typename T>
    struct Rot {
        T s;
        T c;
    };

    template <typename T>
    Rot<T> make_rot(T angle) {
        return {std::sin(angle), std::cos(angle)};
    }

    template <typename T>
    constexpr Vec2<T> mul(Rot<T> q, Vec2<T> v) { return {q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y}; }

    template <typename T>
    struct Vec3 {
        T x;
        T y;
        T z;
    };

    template <typename T>
    constexpr Vec3<T>& operator+=(Vec3<T>& a, Vec3<T> b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
    template <typename T>
    constexpr Vec3<T> operator-(Vec3<T> a) { return {-a.x, -a.y, -a.z}; }
    template <typename T>
    constexpr T dot(Vec3<T> a, Vec3<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    template <typename T>
    constexpr Vec3<T> cross(Vec3<T> a, Vec3<T> b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    // b2Mat33 almacenada por columnas.
    template <typename T>
    struct Mat33 {
        Vec3<T> ex;
        Vec3<T> ey;
        Vec3<T> ez;
    };

    // b2Mat33::Solve33: resuelve A x = b por la regla de Cramer.
    template <typename T>
    constexpr Vec3<T> solve33(const Mat33<T>& m, Vec3<T> b) {
        T det = dot(m.ex, cross(m.ey, m.ez));
        if (det != 0) {
            det = 1 / det;
        }
        return {det * dot(b, cross(m.ey, m.ez)),
                det * dot(m.ex, cross(b, m.ez)),
                det * dot(m.ex, cross(m.ey, b))};
    }

    // b2Mat33::Solve22 (bloque 2x2 superior izquierdo) y b2Mat22::Solve: misma fórmula.
    template <typename T>
    constexpr Vec2<T> solve22(T a11, T a12, T a21, T a22, Vec2<T> b) {
        T det = a11 * a22 - a12 * a21;
        if (det != 0) {
            det = 1 / det;
        }
        return {det * (a22 * b.x - a12 * b.y), det * (a11 * b.y - a21 * b.x)};
    }

}
