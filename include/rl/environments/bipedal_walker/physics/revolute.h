// Junta revoluta con motor y límites, portada de b2RevoluteJoint (Box2D 2.3.x, licencia zlib).
// Copyright (c) 2006-2011 Erin Catto http://www.box2d.org
//
// Diferencias deliberadas con Box2D, todas irrelevantes para este entorno:
//  - Sin warm-starting: los impulsos acumulados valen 0 al inicio de cada paso, así que la
//    junta no tiene estado entre pasos (el estado de límite se recalcula cada vez).
//  - Motor y límite siempre activados; rotación nunca fija; los límites nunca son iguales
//    (equalLimits). Se eliminan esas ramas.
#pragma once

#include "body_mass.h"
#include "settings.h"

namespace bipedal_walker::physics {

    // Posición del centro de masa (c), ángulo (a) y velocidades de un cuerpo. Es lo mismo que
    // b2Position + b2Velocity del solver de Box2D.
    template <typename T>
    struct BodyState {
        Vec2<T> c;
        T a;
        Vec2<T> v;
        T w;
    };

    template <typename T>
    struct RevoluteJointDef {
        int a;
        int b;
        Vec2<T> local_anchor_a;
        Vec2<T> local_anchor_b;
        T lower_angle;
        T upper_angle;
        // Ángulo relativo (B - A) en el que la junta vale 0. pybox2d lo fija al crear la junta
        // con bodyA/bodyB como bodyB.angle - bodyA.angle: en las caderas es +-0.05 (las piernas
        // nacen inclinadas) y así la cadera observada parte de 0.
        T reference_angle;
    };

    enum class LimitState { Inactive, AtLower, AtUpper };

    // Datos que la junta calcula al inicio del paso y reutiliza en cada iteración.
    template <typename T>
    struct RevoluteJointScratch {
        Vec2<T> r_a;
        Vec2<T> r_b;
        Mat33<T> mass;
        T motor_mass;
        LimitState limit_state;
        Vec3<T> impulse;
        T motor_impulse;
        T motor_speed;
        T max_motor_torque;
    };

    template <typename T>
    void joint_init_velocity(const RevoluteJointDef<T>& d, RevoluteJointScratch<T>& s,
                             const BodyState<T>* bodies, const BodyMass<T>& ma, const BodyMass<T>& mb) {
        T aA = bodies[d.a].a;
        T aB = bodies[d.b].a;
        Rot<T> qA = make_rot(aA);
        Rot<T> qB = make_rot(aB);

        s.r_a = mul(qA, d.local_anchor_a - ma.local_center);
        s.r_b = mul(qB, d.local_anchor_b - mb.local_center);

        T mA = ma.inv_mass, mB = mb.inv_mass;
        T iA = ma.inv_I, iB = mb.inv_I;

        s.mass.ex.x = mA + mB + s.r_a.y * s.r_a.y * iA + s.r_b.y * s.r_b.y * iB;
        s.mass.ey.x = -s.r_a.y * s.r_a.x * iA - s.r_b.y * s.r_b.x * iB;
        s.mass.ez.x = -s.r_a.y * iA - s.r_b.y * iB;
        s.mass.ex.y = s.mass.ey.x;
        s.mass.ey.y = mA + mB + s.r_a.x * s.r_a.x * iA + s.r_b.x * s.r_b.x * iB;
        s.mass.ez.y = s.r_a.x * iA + s.r_b.x * iB;
        s.mass.ex.z = s.mass.ez.x;
        s.mass.ey.z = s.mass.ez.y;
        s.mass.ez.z = iA + iB;

        s.motor_mass = iA + iB;
        if (s.motor_mass > 0) {
            s.motor_mass = 1 / s.motor_mass;
        }

        T joint_angle = aB - aA - d.reference_angle;
        if (joint_angle <= d.lower_angle) {
            s.limit_state = LimitState::AtLower;
        } else if (joint_angle >= d.upper_angle) {
            s.limit_state = LimitState::AtUpper;
        } else {
            s.limit_state = LimitState::Inactive;
        }

        s.impulse = {0, 0, 0};
        s.motor_impulse = 0;
    }

    template <typename T>
    void joint_solve_velocity(const RevoluteJointDef<T>& d, RevoluteJointScratch<T>& s, BodyState<T>* bodies,
                              const BodyMass<T>& ma, const BodyMass<T>& mb, T dt) {
        Vec2<T> vA = bodies[d.a].v;
        T wA = bodies[d.a].w;
        Vec2<T> vB = bodies[d.b].v;
        T wB = bodies[d.b].w;

        T mA = ma.inv_mass, mB = mb.inv_mass;
        T iA = ma.inv_I, iB = mb.inv_I;

        // Motor: sigue motor_speed con un impulso acotado por dt * max_motor_torque.
        {
            T Cdot = wB - wA - s.motor_speed;
            T impulse = -s.motor_mass * Cdot;
            T old_impulse = s.motor_impulse;
            T max_impulse = dt * s.max_motor_torque;
            s.motor_impulse = clamp(s.motor_impulse + impulse, -max_impulse, max_impulse);
            impulse = s.motor_impulse - old_impulse;

            wA -= iA * impulse;
            wB += iB * impulse;
        }

        if (s.limit_state != LimitState::Inactive) {
            // Restricción de punto (2D) y de límite (1D) resueltas juntas con la matriz 3x3.
            Vec2<T> Cdot1 = vB + cross(wB, s.r_b) - vA - cross(wA, s.r_a);
            T Cdot2 = wB - wA;
            Vec3<T> Cdot = {Cdot1.x, Cdot1.y, Cdot2};

            Vec3<T> impulse = -solve33(s.mass, Cdot);

            // El impulso del límite solo puede empujar hacia dentro (z <= 0 abajo, z >= 0 arriba);
            // si se saliera, se rehace el punto como 2x2 y se cancela el impulso de límite.
            bool clamped;
            T new_impulse = s.impulse.z + impulse.z;
            if (s.limit_state == LimitState::AtLower) {
                clamped = new_impulse < 0;
            } else {
                clamped = new_impulse > 0;
            }

            if (clamped) {
                Vec2<T> rhs = -Cdot1 + s.impulse.z * Vec2<T>{s.mass.ez.x, s.mass.ez.y};
                Vec2<T> reduced = solve22(s.mass.ex.x, s.mass.ey.x, s.mass.ex.y, s.mass.ey.y, rhs);
                impulse.x = reduced.x;
                impulse.y = reduced.y;
                impulse.z = -s.impulse.z;
                s.impulse.x += reduced.x;
                s.impulse.y += reduced.y;
                s.impulse.z = 0;
            } else {
                s.impulse += impulse;
            }

            Vec2<T> P = {impulse.x, impulse.y};

            vA -= mA * P;
            wA -= iA * (cross(s.r_a, P) + impulse.z);

            vB += mB * P;
            wB += iB * (cross(s.r_b, P) + impulse.z);
        } else {
            Vec2<T> Cdot = vB + cross(wB, s.r_b) - vA - cross(wA, s.r_a);
            Vec2<T> impulse = solve22(s.mass.ex.x, s.mass.ey.x, s.mass.ex.y, s.mass.ey.y, -Cdot);

            s.impulse.x += impulse.x;
            s.impulse.y += impulse.y;

            vA -= mA * impulse;
            wA -= iA * cross(s.r_a, impulse);

            vB += mB * impulse;
            wB += iB * cross(s.r_b, impulse);
        }

        bodies[d.a].v = vA;
        bodies[d.a].w = wA;
        bodies[d.b].v = vB;
        bodies[d.b].w = wB;
    }

    // Corrección de posición del punto y del límite. Devuelve true si los errores están dentro
    // de las tolerancias (linear_slop, angular_slop).
    template <typename T>
    bool joint_solve_position(const RevoluteJointDef<T>& d, const RevoluteJointScratch<T>& s, BodyState<T>* bodies,
                              const BodyMass<T>& ma, const BodyMass<T>& mb) {
        using S = Settings<T>;

        Vec2<T> cA = bodies[d.a].c;
        T aA = bodies[d.a].a;
        Vec2<T> cB = bodies[d.b].c;
        T aB = bodies[d.b].a;

        T angular_error = 0;
        T position_error = 0;

        if (s.limit_state != LimitState::Inactive) {
            T angle = aB - aA - d.reference_angle;
            T limit_impulse = 0;

            if (s.limit_state == LimitState::AtLower) {
                T C = angle - d.lower_angle;
                angular_error = -C;

                // Permite un poco de penetración (angular_slop) para evitar trepidación.
                C = clamp(C + S::ANGULAR_SLOP, -S::MAX_ANGULAR_CORRECTION, T(0));
                limit_impulse = -s.motor_mass * C;
            } else {
                T C = angle - d.upper_angle;
                angular_error = C;

                C = clamp(C - S::ANGULAR_SLOP, T(0), S::MAX_ANGULAR_CORRECTION);
                limit_impulse = -s.motor_mass * C;
            }

            aA -= ma.inv_I * limit_impulse;
            aB += mb.inv_I * limit_impulse;
        }

        {
            Rot<T> qA = make_rot(aA);
            Rot<T> qB = make_rot(aB);
            Vec2<T> rA = mul(qA, d.local_anchor_a - ma.local_center);
            Vec2<T> rB = mul(qB, d.local_anchor_b - mb.local_center);

            Vec2<T> C = cB + rB - cA - rA;
            position_error = std::sqrt(C.x * C.x + C.y * C.y);

            T mA = ma.inv_mass, mB = mb.inv_mass;
            T iA = ma.inv_I, iB = mb.inv_I;

            T k_ex_x = mA + mB + iA * rA.y * rA.y + iB * rB.y * rB.y;
            T k_ex_y = -iA * rA.x * rA.y - iB * rB.x * rB.y;
            T k_ey_x = k_ex_y;
            T k_ey_y = mA + mB + iA * rA.x * rA.x + iB * rB.x * rB.x;

            Vec2<T> impulse = -solve22(k_ex_x, k_ey_x, k_ex_y, k_ey_y, C);

            cA -= mA * impulse;
            aA -= iA * cross(rA, impulse);

            cB += mB * impulse;
            aB += iB * cross(rB, impulse);
        }

        bodies[d.a].c = cA;
        bodies[d.a].a = aA;
        bodies[d.b].c = cB;
        bodies[d.b].a = aB;

        return position_error <= S::LINEAR_SLOP && angular_error <= S::ANGULAR_SLOP;
    }

}
