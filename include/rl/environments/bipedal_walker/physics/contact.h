// Solver de contactos, portado de b2ContactSolver (Box2D 2.3.x, licencia zlib, Erin Catto).
//
// Todos los contactos del caminante son (arista estática del terreno) x (cuerpo dinámico), con la
// arista como cuerpo A. Un cuerpo A estático tiene masa, inercia y velocidad nulas, así que los
// términos de A se anulan exactamente y se omiten. Sin warm-starting: los impulsos acumulados
// empiezan en 0 en cada paso. Restitución 0, luego velocityBias = 0 siempre.
#pragma once

#include "manifold.h"
#include "revolute.h"

namespace bipedal_walker::physics {

    template <typename T>
    struct ContactPoint {
        Vec2<T> r_b;  // punto de contacto respecto al centro de masa del cuerpo B
        T normal_impulse;
        T tangent_impulse;
        T normal_mass;
        T tangent_mass;
    };

    template <typename T>
    struct ContactConstraint {
        int body;  // índice del cuerpo B (dinámico); el A es el suelo
        T friction;
        Manifold<T> manifold;

        Vec2<T> normal;
        ContactPoint<T> points[2];
        // Bloque de 2 puntos: K y su inversa (columnas ex, ey).
        T k11, k12, k22;
        T nm_ex_x, nm_ex_y, nm_ey_x, nm_ey_y;
        int point_count;  // del solver de velocidad: baja a 1 si el bloque está mal condicionado
    };

    // Puntos y normal del contacto en coordenadas de mundo (b2WorldManifold::Initialize) para
    // xfA = identidad y xfB = (q, p). La normal apunta de A (suelo) a B.
    template <typename T>
    void world_manifold(const Manifold<T>& m, Rot<T> q, Vec2<T> p, Vec2<T>& normal, Vec2<T> points[2]) {
        using S = Settings<T>;
        const T radius_a = S::POLYGON_RADIUS;
        const T radius_b = S::POLYGON_RADIUS;

        if (m.type == Manifold<T>::Type::FaceA) {
            normal = m.local_normal;
            Vec2<T> plane_point = m.local_point;
            for (int i = 0; i < m.count; ++i) {
                Vec2<T> clip_point = mul(q, m.points[i]) + p;
                Vec2<T> cA = clip_point + (radius_a - dot(clip_point - plane_point, normal)) * normal;
                Vec2<T> cB = clip_point - radius_b * normal;
                points[i] = T(0.5) * (cA + cB);
            }
        } else {
            normal = mul(q, m.local_normal);
            Vec2<T> plane_point = mul(q, m.local_point) + p;
            for (int i = 0; i < m.count; ++i) {
                Vec2<T> clip_point = m.points[i];
                Vec2<T> cB = clip_point + (radius_b - dot(clip_point - plane_point, normal)) * normal;
                Vec2<T> cA = clip_point - radius_a * normal;
                points[i] = T(0.5) * (cA + cB);
            }
            normal = -normal;
        }
    }

    // Origen del cuerpo a partir de su centro de masa: p = c - R(a) * centro_local.
    template <typename T>
    inline Vec2<T> body_position(const BodyState<T>& b, const BodyMass<T>& mass, Rot<T> q) {
        return b.c - mul(q, mass.local_center);
    }

    // InitializeVelocityConstraints (restitución 0: velocityBias = 0).
    template <typename T>
    void contact_init_velocity(ContactConstraint<T>& vc, const BodyState<T>& body, const BodyMass<T>& mass) {
        const T mB = mass.inv_mass;
        const T iB = mass.inv_I;
        const Manifold<T>& m = vc.manifold;

        Rot<T> q = make_rot(body.a);
        Vec2<T> p = body_position(body, mass, q);

        Vec2<T> world_points[2];
        world_manifold(m, q, p, vc.normal, world_points);

        vc.point_count = m.count;
        for (int j = 0; j < vc.point_count; ++j) {
            ContactPoint<T>& vcp = vc.points[j];
            vcp.normal_impulse = 0;
            vcp.tangent_impulse = 0;
            vcp.r_b = world_points[j] - body.c;

            T rnB = cross(vcp.r_b, vc.normal);
            T k_normal = mB + iB * rnB * rnB;
            vcp.normal_mass = k_normal > 0 ? 1 / k_normal : 0;

            Vec2<T> tangent = cross(vc.normal, T(1));
            T rtB = cross(vcp.r_b, tangent);
            T k_tangent = mB + iB * rtB * rtB;
            vcp.tangent_mass = k_tangent > 0 ? 1 / k_tangent : 0;
        }

        // Bloque de 2 puntos: se resuelve conjuntamente si K está bien condicionada.
        if (vc.point_count == 2) {
            T rn1B = cross(vc.points[0].r_b, vc.normal);
            T rn2B = cross(vc.points[1].r_b, vc.normal);

            T k11 = mB + iB * rn1B * rn1B;
            T k22 = mB + iB * rn2B * rn2B;
            T k12 = mB + iB * rn1B * rn2B;

            constexpr T MAX_CONDITION_NUMBER = 1000;
            if (k11 * k11 < MAX_CONDITION_NUMBER * (k11 * k22 - k12 * k12)) {
                vc.k11 = k11;
                vc.k12 = k12;
                vc.k22 = k22;
                // b2Mat22::GetInverse
                T det = k11 * k22 - k12 * k12;
                if (det != 0) {
                    det = 1 / det;
                }
                vc.nm_ex_x = det * k22;
                vc.nm_ey_x = -det * k12;
                vc.nm_ex_y = -det * k12;
                vc.nm_ey_y = det * k11;
            } else {
                vc.point_count = 1;
            }
        }
    }

    // Aplica el impulso P (en el punto r_b) al cuerpo B: v += mB * P, w += iB * cross(r_b, P).
    template <typename T>
    inline void apply_impulse_b(Vec2<T>& vB, T& wB, T mB, T iB, Vec2<T> r_b, Vec2<T> P) {
        vB += mB * P;
        wB += iB * cross(r_b, P);
    }

    // SolveVelocityConstraints de un contacto: primero fricción, luego normal.
    template <typename T>
    void contact_solve_velocity(ContactConstraint<T>& vc, BodyState<T>& body, const BodyMass<T>& mass) {
        const T mB = mass.inv_mass;
        const T iB = mass.inv_I;

        Vec2<T> vB = body.v;
        T wB = body.w;

        const Vec2<T> normal = vc.normal;
        const Vec2<T> tangent = cross(normal, T(1));
        const T friction = vc.friction;

        // Fricción (tangencial): acotada por friction * impulso normal acumulado.
        for (int j = 0; j < vc.point_count; ++j) {
            ContactPoint<T>& vcp = vc.points[j];

            Vec2<T> dv = vB + cross(wB, vcp.r_b);
            T vt = dot(dv, tangent);
            T lambda = vcp.tangent_mass * (-vt);

            T max_friction = friction * vcp.normal_impulse;
            T new_impulse = clamp(vcp.tangent_impulse + lambda, -max_friction, max_friction);
            lambda = new_impulse - vcp.tangent_impulse;
            vcp.tangent_impulse = new_impulse;

            apply_impulse_b(vB, wB, mB, iB, vcp.r_b, lambda * tangent);
        }

        if (vc.point_count == 1) {
            ContactPoint<T>& vcp = vc.points[0];

            Vec2<T> dv = vB + cross(wB, vcp.r_b);
            T vn = dot(dv, normal);
            T lambda = -vcp.normal_mass * vn;

            T new_impulse = max(vcp.normal_impulse + lambda, T(0));
            lambda = new_impulse - vcp.normal_impulse;
            vcp.normal_impulse = new_impulse;

            apply_impulse_b(vB, wB, mB, iB, vcp.r_b, lambda * normal);
        } else {
            // Bloque de 2 puntos: LCP de 2 variables, se prueban los 4 casos de conjunto activo.
            ContactPoint<T>& cp1 = vc.points[0];
            ContactPoint<T>& cp2 = vc.points[1];
            Vec2<T> a = {cp1.normal_impulse, cp2.normal_impulse};

            Vec2<T> dv1 = vB + cross(wB, cp1.r_b);
            Vec2<T> dv2 = vB + cross(wB, cp2.r_b);

            T vn1 = dot(dv1, normal);
            T vn2 = dot(dv2, normal);

            Vec2<T> b = {vn1, vn2};
            // b' = b - K a
            b.x -= vc.k11 * a.x + vc.k12 * a.y;
            b.y -= vc.k12 * a.x + vc.k22 * a.y;

            auto apply = [&](Vec2<T> x) {
                Vec2<T> d = x - a;
                Vec2<T> P1 = d.x * normal;
                Vec2<T> P2 = d.y * normal;
                vB += mB * (P1 + P2);
                wB += iB * (cross(cp1.r_b, P1) + cross(cp2.r_b, P2));
                cp1.normal_impulse = x.x;
                cp2.normal_impulse = x.y;
            };

            for (;;) {
                // Caso 1: ambos puntos activos (vn1 = vn2 = 0).
                Vec2<T> x = -Vec2<T>{vc.nm_ex_x * b.x + vc.nm_ey_x * b.y, vc.nm_ex_y * b.x + vc.nm_ey_y * b.y};
                if (x.x >= 0 && x.y >= 0) {
                    apply(x);
                    break;
                }

                // Caso 2: solo el punto 1 activo (x2 = 0, vn1 = 0).
                x.x = -cp1.normal_mass * b.x;
                x.y = 0;
                vn2 = vc.k12 * x.x + b.y;
                if (x.x >= 0 && vn2 >= 0) {
                    apply(x);
                    break;
                }

                // Caso 3: solo el punto 2 activo (x1 = 0, vn2 = 0).
                x.x = 0;
                x.y = -cp2.normal_mass * b.y;
                vn1 = vc.k12 * x.y + b.x;
                if (x.y >= 0 && vn1 >= 0) {
                    apply(x);
                    break;
                }

                // Caso 4: ninguno activo.
                x.x = 0;
                x.y = 0;
                vn1 = b.x;
                vn2 = b.y;
                if (vn1 >= 0 && vn2 >= 0) {
                    apply(x);
                    break;
                }

                // Sin solución: se dejan los impulsos como están.
                break;
            }
        }

        body.v = vB;
        body.w = wB;
    }

    // SolvePositionConstraints de un contacto. Devuelve la separación mínima encontrada
    // (mínimo con `min_separation`), que el llamador compara con -3 * linear_slop.
    template <typename T>
    T contact_solve_position(const ContactConstraint<T>& vc, BodyState<T>& body, const BodyMass<T>& mass,
                             T min_separation) {
        using S = Settings<T>;
        const T mB = mass.inv_mass;
        const T iB = mass.inv_I;
        const Manifold<T>& m = vc.manifold;

        Vec2<T> cB = body.c;
        T aB = body.a;

        // El solver de posición usa todos los puntos del manifold, aunque el de velocidad
        // haya bajado a 1 (pc->pointCount frente a vc->pointCount).
        for (int j = 0; j < m.count; ++j) {
            Rot<T> q = make_rot(aB);
            Vec2<T> p = cB - mul(q, mass.local_center);

            Vec2<T> normal, point;
            T separation;
            if (m.type == Manifold<T>::Type::FaceA) {
                normal = m.local_normal;
                Vec2<T> plane_point = m.local_point;
                Vec2<T> clip_point = mul(q, m.points[j]) + p;
                separation = dot(clip_point - plane_point, normal) - S::POLYGON_RADIUS - S::POLYGON_RADIUS;
                point = clip_point;
            } else {
                normal = mul(q, m.local_normal);
                Vec2<T> plane_point = mul(q, m.local_point) + p;
                Vec2<T> clip_point = m.points[j];
                separation = dot(clip_point - plane_point, normal) - S::POLYGON_RADIUS - S::POLYGON_RADIUS;
                point = clip_point;
                normal = -normal;
            }

            Vec2<T> rB = point - cB;

            min_separation = min(min_separation, separation);

            // Corrección de Baumgarte, con penetración permitida (linear_slop) y tope por paso.
            T C = clamp(S::BAUMGARTE * (separation + S::LINEAR_SLOP), -S::MAX_LINEAR_CORRECTION, T(0));

            T rnB = cross(rB, normal);
            T K = mB + iB * rnB * rnB;

            T impulse = K > 0 ? -C / K : 0;
            Vec2<T> P = impulse * normal;

            cB += mB * P;
            aB += iB * cross(rB, P);
        }

        body.c = cB;
        body.a = aB;
        return min_separation;
    }

}
