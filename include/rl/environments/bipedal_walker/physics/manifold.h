// Colisión arista-polígono, portada de b2CollideEdgeAndPolygon / b2EPCollider (Box2D 2.3.x,
// licencia zlib, Copyright (c) 2007-2009 Erin Catto http://www.box2d.org).
//
// Simplificaciones, válidas porque el terreno es un conjunto de aristas simples y estáticas
// (sin vértices fantasma: hasVertex0 = hasVertex3 = false) en el origen del mundo (transformada
// identidad):
//  - Solo se porta la rama de b2EPCollider::Collide sin aristas adyacentes.
//  - No se guardan los ids de característica (solo sirven para el warm-starting).
#pragma once

#include "polygon.h"
#include "settings.h"

namespace bipedal_walker::physics {

    template <typename T>
    struct Manifold {
        enum class Type { FaceA, FaceB };
        Type type = Type::FaceA;
        Vec2<T> local_normal = {0, 0};
        Vec2<T> local_point = {0, 0};
        Vec2<T> points[2] = {{0, 0}, {0, 0}};  // puntos de contacto en el marco local del cuerpo B
        int count = 0;
    };

    // b2ClipSegmentToLine sin ids: recorta el segmento v_in[0..1] contra el semiplano
    // dot(normal, x) <= offset y devuelve cuántos puntos quedan.
    template <typename T>
    int clip_segment_to_line(Vec2<T> v_out[2], const Vec2<T> v_in[2], Vec2<T> normal, T offset) {
        int num_out = 0;

        T distance0 = dot(normal, v_in[0]) - offset;
        T distance1 = dot(normal, v_in[1]) - offset;

        if (distance0 <= 0) v_out[num_out++] = v_in[0];
        if (distance1 <= 0) v_out[num_out++] = v_in[1];

        // Puntos a lados distintos del plano: añadir la intersección.
        if (distance0 * distance1 < 0) {
            T interp = distance0 / (distance0 - distance1);
            v_out[num_out] = v_in[0] + interp * (v_in[1] - v_in[0]);
            ++num_out;
        }
        return num_out;
    }

    // Colisiona la arista v1-v2 (mundo) con `polygon` cuyo cuerpo tiene transformada (q, p).
    // El polígono es el shape B; la arista, el A (estática).
    template <typename T, int CAPACITY>
    void collide_edge_polygon(Manifold<T>& manifold, Vec2<T> v1, Vec2<T> v2,
                              const Polygon<T, CAPACITY>& polygon, Rot<T> q, Vec2<T> p) {
        using S = Settings<T>;
        constexpr T RADIUS = 2 * S::POLYGON_RADIUS;

        manifold.count = 0;

        // Con xfA = identidad, m_xf = MulT(xfA, xfB) = xfB.
        auto xf_mul = [&](Vec2<T> v) {
            return Vec2<T>{(q.c * v.x - q.s * v.y) + p.x, (q.s * v.x + q.c * v.y) + p.y};
        };
        auto xf_mul_t = [&](Vec2<T> v) {
            T px = v.x - p.x;
            T py = v.y - p.y;
            return Vec2<T>{q.c * px + q.s * py, -q.s * px + q.c * py};
        };

        Vec2<T> centroid_b = xf_mul(polygon.centroid);

        Vec2<T> edge1 = v2 - v1;
        normalize(edge1);
        Vec2<T> normal1 = {edge1.y, -edge1.x};
        T offset1 = dot(normal1, centroid_b - v1);

        // Cara frontal o trasera, y límites de la normal (rama sin vértices adyacentes).
        bool front = offset1 >= 0;
        Vec2<T> normal, lower_limit, upper_limit;
        if (front) {
            normal = normal1;
            lower_limit = -normal1;
            upper_limit = -normal1;
        } else {
            normal = -normal1;
            lower_limit = normal1;
            upper_limit = normal1;
        }

        // Polígono B en el marco de la arista.
        Vec2<T> vertices[CAPACITY];
        Vec2<T> normals[CAPACITY];
        const int count = polygon.count;
        for (int i = 0; i < count; ++i) {
            vertices[i] = xf_mul(polygon.vertices[i]);
            normals[i] = mul(q, polygon.normals[i]);
        }

        // Separación según la normal de la arista.
        T edge_separation = std::numeric_limits<T>::max();
        for (int i = 0; i < count; ++i) {
            T s = dot(normal, vertices[i] - v1);
            if (s < edge_separation) {
                edge_separation = s;
            }
        }
        if (edge_separation > RADIUS) {
            return;
        }

        // Separación según las normales del polígono.
        enum class AxisType { Unknown, EdgeA, EdgeB };
        AxisType poly_type = AxisType::Unknown;
        int poly_index = -1;
        T poly_separation = -std::numeric_limits<T>::max();
        {
            Vec2<T> perp = {-normal.y, normal.x};
            for (int i = 0; i < count; ++i) {
                Vec2<T> n = -normals[i];

                T s1 = dot(n, vertices[i] - v1);
                T s2 = dot(n, vertices[i] - v2);
                T s = min(s1, s2);

                if (s > RADIUS) {
                    // Eje separador: sin contacto.
                    return;
                }

                // Solo se aceptan ejes dentro del rango de normales admisible.
                if (dot(n, perp) >= 0) {
                    if (dot(n - upper_limit, normal) < -S::ANGULAR_SLOP) {
                        continue;
                    }
                } else {
                    if (dot(n - lower_limit, normal) < -S::ANGULAR_SLOP) {
                        continue;
                    }
                }

                if (s > poly_separation) {
                    poly_type = AxisType::EdgeB;
                    poly_index = i;
                    poly_separation = s;
                }
            }
        }

        // Elegir el eje principal: se prefiere la arista salvo que el polígono sea claramente mejor.
        constexpr T RELATIVE_TOL = static_cast<T>(0.98f);
        constexpr T ABSOLUTE_TOL = static_cast<T>(0.001f);
        bool primary_is_edge;
        if (poly_type == AxisType::Unknown) {
            primary_is_edge = true;
        } else if (poly_separation > RELATIVE_TOL * edge_separation + ABSOLUTE_TOL) {
            primary_is_edge = false;
        } else {
            primary_is_edge = true;
        }

        // Cara de referencia y segmento incidente.
        Vec2<T> ie[2];
        struct ReferenceFace {
            Vec2<T> v1, v2, normal, side_normal1, side_normal2;
            T side_offset1, side_offset2;
        } rf;
        int rf_i1;

        if (primary_is_edge) {
            manifold.type = Manifold<T>::Type::FaceA;

            // Cara del polígono más antiparalela a la normal de la arista.
            int best_index = 0;
            T best_value = dot(normal, normals[0]);
            for (int i = 1; i < count; ++i) {
                T value = dot(normal, normals[i]);
                if (value < best_value) {
                    best_value = value;
                    best_index = i;
                }
            }
            int i1 = best_index;
            int i2 = i1 + 1 < count ? i1 + 1 : 0;
            ie[0] = vertices[i1];
            ie[1] = vertices[i2];

            if (front) {
                rf.v1 = v1;
                rf.v2 = v2;
                rf.normal = normal1;
            } else {
                rf.v1 = v2;
                rf.v2 = v1;
                rf.normal = -normal1;
            }
            rf_i1 = 0;
        } else {
            manifold.type = Manifold<T>::Type::FaceB;
            ie[0] = v1;
            ie[1] = v2;
            rf_i1 = poly_index;
            int rf_i2 = rf_i1 + 1 < count ? rf_i1 + 1 : 0;
            rf.v1 = vertices[rf_i1];
            rf.v2 = vertices[rf_i2];
            rf.normal = normals[rf_i1];
        }

        rf.side_normal1 = {rf.normal.y, -rf.normal.x};
        rf.side_normal2 = -rf.side_normal1;
        rf.side_offset1 = dot(rf.side_normal1, rf.v1);
        rf.side_offset2 = dot(rf.side_normal2, rf.v2);

        Vec2<T> clip_points1[2];
        Vec2<T> clip_points2[2];
        int np = clip_segment_to_line(clip_points1, ie, rf.side_normal1, rf.side_offset1);
        if (np < 2) {
            return;
        }
        np = clip_segment_to_line(clip_points2, clip_points1, rf.side_normal2, rf.side_offset2);
        if (np < 2) {
            return;
        }

        if (primary_is_edge) {
            manifold.local_normal = rf.normal;
            manifold.local_point = rf.v1;
        } else {
            manifold.local_normal = polygon.normals[rf_i1];
            manifold.local_point = polygon.vertices[rf_i1];
        }

        int point_count = 0;
        for (int i = 0; i < 2; ++i) {
            T separation = dot(rf.normal, clip_points2[i] - rf.v1);
            if (separation <= RADIUS) {
                manifold.points[point_count] = primary_is_edge ? xf_mul_t(clip_points2[i]) : clip_points2[i];
                ++point_count;
            }
        }
        manifold.count = point_count;
    }

    // Colisión polígono-polígono, portada de b2CollidePolygons / b2FindMaxSeparation /
    // b2FindIncidentEdge (Box2D 2.3.x, licencia zlib, Erin Catto). El polígono A (los obstáculos
    // del terreno hardcore) ya está en coordenadas de mundo, transformada identidad; el B (una
    // pierna o el casco) tiene rotación q y traslación p, igual que en collide_edge_polygon.
    // No se guardan ids de característica (solo sirven para warm-starting, que este port no usa).
    template <typename T, int CAP_A, int CAP_B>
    void collide_polygons(Manifold<T>& manifold, const Polygon<T, CAP_A>& poly_a,
                          const Polygon<T, CAP_B>& poly_b, Rot<T> qb, Vec2<T> pb) {
        using S = Settings<T>;
        constexpr T RADIUS = 2 * S::POLYGON_RADIUS;

        manifold.count = 0;

        // Vértices y normales de B en coordenadas de mundo (A ya lo está).
        Vec2<T> vb[CAP_B], nb[CAP_B];
        for (int i = 0; i < poly_b.count; ++i) {
            vb[i] = mul(qb, poly_b.vertices[i]) + pb;
            nb[i] = mul(qb, poly_b.normals[i]);
        }

        // Separación máxima: b2FindMaxSeparation (Box2D 2.3.x). No es una búsqueda exhaustiva:
        // parte de la normal más alineada con la dirección al centroide ajeno y trepa al vecino
        // con mayor separación hasta que ninguno mejora. Es un óptimo LOCAL, no necesariamente el
        // global; hay que reproducirlo tal cual (en vez de un máximo por fuerza bruta) porque en
        // empates o casi-empates Box2D puede quedarse en una arista de referencia distinta a la
        // de separación máxima real, lo que cambia el manifiesto entero, no solo su precisión.
        auto edge_separation = [](const Vec2<T>* v1, const Vec2<T>* n1, int e1, const Vec2<T>* v2, int count2) {
            T min_dot = std::numeric_limits<T>::max();
            int index = 0;
            for (int i = 0; i < count2; ++i) {
                T d = dot(v2[i], n1[e1]);
                if (d < min_dot) { min_dot = d; index = i; }
            }
            return dot(v2[index] - v1[e1], n1[e1]);
        };

        auto find_max_separation = [&](const Vec2<T>* v1, const Vec2<T>* n1, int count1, Vec2<T> centroid1,
                                       const Vec2<T>* v2, int count2, Vec2<T> centroid2, int& edge_out) {
            Vec2<T> d = centroid2 - centroid1;
            int edge = 0;
            T max_dot = -std::numeric_limits<T>::max();
            for (int i = 0; i < count1; ++i) {
                T dt = dot(n1[i], d);
                if (dt > max_dot) { max_dot = dt; edge = i; }
            }

            T s = edge_separation(v1, n1, edge, v2, count2);
            int prev_edge = edge - 1 >= 0 ? edge - 1 : count1 - 1;
            T s_prev = edge_separation(v1, n1, prev_edge, v2, count2);
            int next_edge = edge + 1 < count1 ? edge + 1 : 0;
            T s_next = edge_separation(v1, n1, next_edge, v2, count2);

            int best_edge;
            T best_separation;
            int increment;
            if (s_prev > s && s_prev > s_next) {
                increment = -1; best_edge = prev_edge; best_separation = s_prev;
            } else if (s_next > s) {
                increment = 1; best_edge = next_edge; best_separation = s_next;
            } else {
                edge_out = edge;
                return s;
            }

            for (;;) {
                edge = increment == -1 ? (best_edge - 1 >= 0 ? best_edge - 1 : count1 - 1)
                                       : (best_edge + 1 < count1 ? best_edge + 1 : 0);
                s = edge_separation(v1, n1, edge, v2, count2);
                if (s > best_separation) {
                    best_edge = edge; best_separation = s;
                } else {
                    break;
                }
            }
            edge_out = best_edge;
            return best_separation;
        };

        const Vec2<T> centroid_a = poly_a.centroid;  // ya en mundo (obstáculo estático, identidad)
        const Vec2<T> centroid_b = mul(qb, poly_b.centroid) + pb;

        int edge_a = 0, edge_b = 0;

        T sep_a = find_max_separation(poly_a.vertices, poly_a.normals, poly_a.count, centroid_a, vb, poly_b.count, centroid_b, edge_a);
        if (sep_a > RADIUS) return;
        T sep_b = find_max_separation(vb, nb, poly_b.count, centroid_b, poly_a.vertices, poly_a.count, centroid_a, edge_b);
        if (sep_b > RADIUS) return;

        constexpr T RELATIVE_TOL = static_cast<T>(0.98f);
        constexpr T ABSOLUTE_TOL = static_cast<T>(0.001f);
        bool flip = sep_b > RELATIVE_TOL * sep_a + ABSOLUTE_TOL;

        // poly1 = referencia, poly2 = incidente.
        const Vec2<T>* v1 = flip ? vb : poly_a.vertices;
        const Vec2<T>* n1 = flip ? nb : poly_a.normals;
        int count1 = flip ? poly_b.count : poly_a.count;
        int edge1 = flip ? edge_b : edge_a;
        const Vec2<T>* v2 = flip ? poly_a.vertices : vb;
        const Vec2<T>* n2 = flip ? poly_a.normals : nb;
        int count2 = flip ? poly_a.count : poly_b.count;
        manifold.type = flip ? Manifold<T>::Type::FaceB : Manifold<T>::Type::FaceA;

        // Arista incidente de poly2: la más antiparalela a la normal de referencia.
        int inc = 0;
        {
            T min_dot = std::numeric_limits<T>::max();
            for (int i = 0; i < count2; ++i) {
                T d = dot(n1[edge1], n2[i]);
                if (d < min_dot) { min_dot = d; inc = i; }
            }
        }
        int inc2 = inc + 1 < count2 ? inc + 1 : 0;
        Vec2<T> incident_edge[2] = {v2[inc], v2[inc2]};

        int iv1 = edge1;
        int iv2 = edge1 + 1 < count1 ? edge1 + 1 : 0;
        Vec2<T> v11 = v1[iv1];
        Vec2<T> v12 = v1[iv2];
        Vec2<T> tangent = v12 - v11;
        normalize(tangent);
        Vec2<T> normal = cross(tangent, T(1));
        Vec2<T> plane_point = T(0.5) * (v11 + v12);

        T front_offset = dot(normal, v11);
        T side_offset1 = -dot(tangent, v11) + RADIUS;
        T side_offset2 = dot(tangent, v12) + RADIUS;

        Vec2<T> clip1[2], clip2[2];
        if (clip_segment_to_line(clip1, incident_edge, -tangent, side_offset1) < 2) return;
        if (clip_segment_to_line(clip2, clip1, tangent, side_offset2) < 2) return;

        manifold.local_normal = normal;
        manifold.local_point = plane_point;

        // Los puntos de contacto se guardan en el marco local del polígono INCIDENTE (poly2), como
        // en b2CollidePolygons (cp->localPoint = MulT(xf2, clipPoints2[i])), no siempre en el de B:
        //   flip=false (type FaceA, referencia=A, incidente=B): aplicar MulT(qb, pb) — B-local.
        //   flip=true  (type FaceB, referencia=B, incidente=A): A es estático e identidad, así que
        //     su marco local es el mundo — no se transforma nada (igual que collide_edge_polygon
        //     en su rama !primary_is_edge).
        int point_count = 0;
        for (int i = 0; i < 2; ++i) {
            T separation = dot(normal, clip2[i]) - front_offset;
            if (separation <= RADIUS) {
                if (flip) {
                    manifold.points[point_count++] = clip2[i];
                } else {
                    Vec2<T> local = clip2[i] - pb;
                    manifold.points[point_count++] = {qb.c * local.x + qb.s * local.y, -qb.s * local.x + qb.c * local.y};
                }
            }
        }
        manifold.count = point_count;
    }

}
