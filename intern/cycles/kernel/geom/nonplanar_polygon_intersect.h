/* SPDX-FileCopyrightText: 2014-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Nonplanar polygon/ray intersections.
 *
 */

#pragma once

#include "kernel/sample/lcg.h"
#include "util/debug.h"
#include <array>

CCL_NAMESPACE_BEGIN

// TODO: move back to math utils?
// solid_angle_formula
// from DNA_modifier_types.h
// MOD_HARNACK_TRIANGULATE = 0,
// MOD_HARNACK_PREQUANTUM = 1,
// MOD_HARNACK_GAUSS_BONNET = 2,
template<typename T>
ccl_device bool ray_nonplanar_polygon_intersect_T(const float3 ray_Pf,
                                                  const float3 ray_Df,
                                                  const float ray_tmin,
                                                  const float ray_tmax,
                                                  const packed_float3 *pts,
                                                  const size_t N,
                                                  const float epsilonf,
                                                  const float levelsetf,
                                                  const int solid_angle_formula,
                                                  const bool use_grad_termination,
                                                  ccl_private float *isect_u,
                                                  ccl_private float *isect_v,
                                                  ccl_private float *isect_t)
{
  using T3 = std::array<T, 3>;
  using T4 = std::array<T, 4>;
  auto dot = [](const T3 &a, const T3 &b) -> T { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto cross = [](const T3 &a, const T3 &b) -> T3 {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  };
  auto len_squared = [&](const T3 &a) -> T { return dot(a, a); };
  auto len = [&](const T3 &a) -> T { return sqrt(len_squared(a)); };
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };
  auto diff = [](const T3 &a, const T3 &b) -> T3 {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  };
  auto diff_f = [](const float3 &a, const T3 &b) -> T3 {
    return {(T)a.x - b[0], (T)a.y - b[1], (T)a.z - b[2]};
  };
  // a + s * b
  auto fma = [](const T3 &a, T s, const T3 &b) -> T3 {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
  };
  //============================================================
  //
  //                          Quaternions
  //
  //============================================================
  // find a vector orthogonal to v
  auto orthogonal = [](const T3 &v) -> T3 {
    if (std::abs(v[0]) <= std::abs(v[1]) && std::abs(v[0]) <= std::abs(v[2])) {
      return T3{0., -v[2], v[1]};
    }
    else if (std::abs(v[1]) <= std::abs(v[0]) && std::abs(v[1]) <= std::abs(v[2])) {
      return T3{v[2], 0., -v[0]};
    }
    else {
      return T3{-v[1], v[0], 0.};
    }
  };
  auto mul_s = [](T s, const T3 &v) -> T3 { return {s * v[0], s * v[1], s * v[2]}; };
  auto q_re = [](const T4 &q) -> T { return q[0]; };
  auto q_im = [](const T4 &q) -> T3 { return {q[1], q[2], q[3]}; };
  auto build_T4 = [](T x, const T3 &yzw) -> T4 { return {x, yzw[0], yzw[1], yzw[2]}; };
  auto q_dot = [](const T4 &a, const T4 &b) -> T {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  };
  auto q_mul = [&](const T4 &a, const T4 &b) -> T4 {
    T3 u = mul_s(q_re(a), q_im(b));
    T3 v = mul_s(q_re(b), q_im(a));
    T3 w = cross(q_im(a), q_im(b));
    return {q_re(a) * q_re(b) - dot(q_im(a), q_im(b)),
            u[0] + v[0] + w[0],
            u[1] + v[1] + w[1],
            u[2] + v[2] + w[2]};
  };
  auto q_conj = [](const T4 &q) -> T4 { return {q[0], -q[1], -q[2], -q[3]}; };
  auto q_div_s = [](const T4 &q, T s) -> T4 { return {q[0] / s, q[1] / s, q[2] / s, q[3] / s}; };
  auto q_inv = [&](const T4 &q) -> T4 { return q_div_s(q_conj(q), q_dot(q, q)); };
  auto q_div = [&](const T4 &a, const T4 &b) -> T4 { return q_mul(a, q_inv(b)); };

  // dihedral of two points on the unit sphere, as defined by Chern & Ishida
  // https://arxiv.org/abs/2303.14555
  auto dihedral = [&](const T3 &p1, const T3 &p2) -> T4 {
    // https://stackoverflow.com/a/11741520
    T lengthProduct = len(p1) * len(p2);

    if (std::abs(dot(p1, p2) / lengthProduct + 1.) < 0.0001) {
      // antiparallel vectors
      return build_T4(0., orthogonal(p1));
    }

    // can skip normalization since we don't care about magnitude
    return build_T4(dot(p1, p2) + lengthProduct, cross(p1, p2));
  };

  // arg(\bar{q2} q1) as defined by Chern & Ishida https://arxiv.org/abs/2303.14555
  auto fiberArg = [&](const T4 &q1, const T4 &q2) -> T {
    T4 s = q_mul(q_conj(q2), q1);
    return atan2(s[1], s[0]);
  };

  int max_iterations = 1500;
  T epsilon = (T)epsilonf;
  T levelset = (T)levelsetf;
  T shift = 4. * M_PI;

  if (N < 3)
    return false;

  T3 ray_P = from_float3(ray_Pf);
  T3 ray_D = from_float3(ray_Df);
  T3 center = from_float3(pts[N]);

  // TODO: try doubles?
  // Fractional mod function that behaves the same as in GLSL
  auto glsl_mod = [](T x, T y) -> T { return x - y * std::floor(x / y); };

  auto close_to_zero = [&](T ang, const T3 &grad) -> bool {
    // Check if an angle is within epsilon of 0 (or 4π)
    T dis = fmin(ang, (T)(4 * M_PI) - ang);
    T tolScaling = use_grad_termination ? len(grad) : 1;
    return dis < epsilon * tolScaling;
  };

  auto distance_to_boundary = [&](const T3 &x, T3 *closest_point) -> T {
    // TODO maybe replace with numeric_limits<T>::infinity()?
    const T infinity = 100000.;
    T min_d2 = infinity;

    // compute closest distance to each polygon line segment
    for (int i = 0; i < N; i++) {
      T3 p1 = from_float3(pts[i]);
      T3 p2 = from_float3(pts[(i + 1) % N]);
      T3 m = diff(p2, p1);
      T3 v = diff(x, p1);
      // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
      T t = fmin(fmax(dot(m, v) / dot(m, m), 0.), 1.);
      T d2 = len_squared(fma(v, -t, m));
      // if closestPoint is not null, update it to track closest point
      if (closest_point && d2 < min_d2)
        *closest_point = fma(p1, t, m);
      min_d2 = fmin(min_d2, d2);
    }

    return std::sqrt(min_d2);
  };

  // Computes a conservative step size via Harnack bounds, using the value
  // fx at the current point, the radius R of a ball over which the function is
  // harmonic, the values lo_bound and up_bound of the closest level sets above/
  // below the current value, and a shift that makes the harmonic function
  // positive within this ball.
  auto get_max_step = [&](T fx, T R, T lo_bound, T up_bound, T shift) -> T {
    T w = (fx + shift) / (up_bound + shift);
    T v = (fx + shift) / (lo_bound + shift);
    T lo_r = -R / 2 * (v + 2 - std::sqrt(v * v + 8 * v));
    T up_r = R / 2 * (w + 2 - std::sqrt(w * w + 8 * w));

    return std::min(lo_r, up_r);
  };

  auto triangulated_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N + 1);
    std::vector<T> Lp;
    Lp.reserve(N + 1);
    for (int i = 0; i < N + 1; i++) {  // center = pts[N]
      xp.push_back(diff_f(pts[i], x));
      Lp.push_back(len(xp[i]));
    }

    // Iterate over triangles used to triangulate the polygon
    std::complex<T> running_angle{1., 0.};
    grad = T3{0., 0., 0.};
    for (int i = 0; i < N; i++) {
      int a = i;
      int b = (i + 1) % N;
      int c = N;

      T3 n = cross(xp[a], xp[b]);

      // Add the solid angle of this triangle to the total
      std::complex<T> tri_angle{Lp[a] * Lp[b] * Lp[c] + dot(xp[a], xp[b]) * Lp[c] +
                                    dot(xp[b], xp[c]) * Lp[a] + dot(xp[a], xp[c]) * Lp[b],
                                dot(xp[c], n)};
      running_angle *= tri_angle;

      //== compute gradient
      if (use_grad_termination) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        grad[0] -= n[0] / n2 * scale;
        grad[1] -= n[1] / n2 * scale;
        grad[2] -= n[2] / n2 * scale;
      }
    }
    return 2 * std::arg(running_angle);
  };

  auto prequantum_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N);
    std::vector<T> Lp;
    Lp.reserve(N);
    for (int i = 0; i < N; i++) {
      xp.push_back(diff_f(pts[i], x));
      Lp.push_back(len(xp[i]));
    }

    int start = 0;
    T4 q0 = dihedral(T3{1., 0., 0.}, xp[start]);  // point in fiber of points[start]-x
    T4 qi = q0;
    for (int i = 0; i < N; i++) {
      int a = i;
      int b = (i + 1) % N;
      T4 d = dihedral(xp[a], xp[b]);
      qi = q_mul(d, qi);
    }

    return -2. * fiberArg(q0, qi);
  };

  auto gauss_bonnet_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N);
    std::vector<T> Lp;
    Lp.reserve(N);
    for (int i = 0; i < N; i++) {
      xp.push_back(diff_f(pts[i], x));
      Lp.push_back(len(xp[i]));
    }

    // Iterate over triangles used to triangulate the polygon
    T total_angle = 0.;
    grad = T3{0., 0., 0.};
    for (int i = 0; i < N; i++) {
      int a = (i + N - 1) % N;
      int b = i;
      int c = (i + 1) % N;
      T3 n_prev = cross(xp[a], xp[b]);
      T3 n_next = cross(xp[b], xp[c]);

      total_angle += atan2(dot(xp[b], cross(n_prev, n_next)) / Lp[b], dot(n_prev, n_next));

      //== compute gradient
      if (use_grad_termination) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n_prev);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        grad[0] -= n_prev[0] / n2 * scale;
        grad[1] -= n_prev[1] / n2 * scale;
        grad[2] -= n_prev[2] / n2 * scale;
      }
    }
    return 2. * M_PI - total_angle;
  };

  /* Find intersection with Harnack tracing */

  T t = (T)ray_tmin;
  int iter = 0;
  T lo_bound = 0;
  T hi_bound = 4. * M_PI;

  T ld = len(ray_D);

  static bool exceeded_max = false;
  while (t < (T)ray_tmax) {
    T3 pos = fma(ray_P, t, ray_D);

    // If we've exceeded the maximum number of iterations,
    // print a warning
    if (iter > max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf("Warning: exceeded maximum number of Harnack iterations.\n");
      }

      /* return false; */
    }

    // To get the most aggressive Harnack bound, we first find a
    // representative of the solid angle, shifted by the target level set
    // value, within the range [0,4π).  Only then do we apply the shift.
    // return fmod(angleSum - levelset, 4.f*M_PI) + shift;
    T3 grad;
    T omega = solid_angle_formula == 0 ? triangulated_solid_angle(pos, grad) :
              solid_angle_formula == 1 ? prequantum_solid_angle(pos, grad) :
                                         gauss_bonnet_solid_angle(pos, grad);
    T val = glsl_mod((double)omega - levelset, 4. * M_PI);

    // Calculate the radius of a ball around the current point over which
    // we know the function is harmonic.  An easy way to identify such a
    // ball is to restrict to the sphere touching the closest point on the
    // polygon boundary.
    T3 closestPoint;
    T R = distance_to_boundary(pos, &closestPoint);

    // If we're close enough to the level set, or we've exceeded the
    // maximum number of iterations, assume there's a hit.
    if (close_to_zero(val, grad) || R < epsilon || iter > max_iterations) {
      // if (R < epsilon)   grad = pos - closestPoint; // TODO: this?
      /* std::string info = "hit info | val: " + std::to_string(val) + " | R: " + std::to_string(R)
       * + */
      /*                    " | iter: " + std::to_string(iter) + "\n"; */
      /* std::cout << info; */
      *isect_t = t;
      *isect_u = 0;
      *isect_v = 0;
      return true;
    }
    else {
      /* std::string info = "not hit info | val: " + std::to_string(val) + */
      /*                    ", iter: " + std::to_string(iter) + "\n"; */
      /* std::cout << info; */
    }

    // Compute a conservative step size based on the Harnack bound.
    T r = get_max_step(val, R, lo_bound, hi_bound, shift);
    t += r / ld;
    iter++;
  }

  return false;
}

ccl_device_inline bool nonplanar_polygon_intersect(KernelGlobals kg,
                                                   ccl_private Intersection *isect,
                                                   float3 P,
                                                   float3 dir,
                                                   float tmin,
                                                   float tmax,
                                                   uint visibility,
                                                   int object,
                                                   int prim,
                                                   int prim_addr)
{
  const uint3 polygon_data = kernel_data_fetch(tri_vindex, prim);
  uint polygon_start = polygon_data.x;
  uint N = polygon_data.y;
  uint solid_angle_formula = polygon_data.z;
  // printf("solid_angle_formula: %d\n", solid_angle_formula);
  const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);
  float3 params = pts[N + 1];
  float epsilon = params.x;
  float levelset = params.y * ((float)(4. * M_PI));
  bool use_grad_termination = static_cast<bool>(params.z);

  float u, v, t;
  if (ray_nonplanar_polygon_intersect_T<double>(P,
                                                dir,
                                                tmin,
                                                tmax,
                                                pts,
                                                N,
                                                epsilon,
                                                levelset,
                                                solid_angle_formula,
                                                use_grad_termination,
                                                &u,
                                                &v,
                                                &t))
  {
#ifdef __VISIBILITY_FLAG__
    /* Visibility flag test. we do it here under the assumption
     * that most triangles are culled by node flags.
     */
    if (kernel_data_fetch(prim_visibility, prim_addr) & visibility)
#endif
    {
      isect->object = object;
      isect->prim = prim;
      isect->type = PRIMITIVE_NONPLANAR_POLYGON;
      isect->u = u;
      isect->v = v;
      isect->t = t;
      return true;
    }
  }
  return false;
}

/* Normal on nonplanar polygon. */
// TODO: move to better-named file
// TODO: deduplicate with version in harnack tracing code?

template<typename T>
ccl_device float3 ray_nonplanar_polygon_normal_T(const float3 pf,
                                                 const packed_float3 *pts,
                                                 const size_t N)
{
  using T3 = std::array<T, 3>;
  auto dot = [](const T3 &a, const T3 &b) -> T { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto cross = [](const T3 &a, const T3 &b) -> T3 {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  };
  auto len_squared = [&](const T3 &a) -> T { return dot(a, a); };
  auto len = [&](const T3 &a) -> T { return sqrt(len_squared(a)); };
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };
  auto diff = [](const T3 &a, const T3 &b) -> T3 {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  };
  // a + s * b
  auto fma = [](const T3 &a, T s, const T3 &b) -> T3 {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
  };
  /* auto smoothstep = [](T x, T start, T end) -> T {  // https://docs.gl/sl4/smoothstep */
  /*   // normalize and clamp to [0, 1] */
  /*   x = fmin(fmax(((x - start) / (end - start)), 0), 1); */

  /*   return x * x * (3. - 2. * x); */
  /* }; */

  // find solid angle gradient and closest point on boundary curve
  T3 p = from_float3(pf);
  T3 grad{0, 0, 0};
  /* , closest_point; */

  /* const T infinity = 100000.; */
  /* T min_d2 = infinity; */
  for (int i = 0; i < N; i++) {
    T3 p0 = from_float3(pts[(i + 0) % N]);
    T3 p1 = from_float3(pts[(i + 1) % N]);
    T3 g0 = diff(p0, p);
    T3 g1 = diff(p1, p);

    //== compute gradient
    T3 n = cross(g1, g0);
    T n2 = len_squared(n);
    T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) + (-dot(g0, g1) + dot(g1, g1)) / len(g1));
    grad[0] -= n[0] / n2 * scale;
    grad[1] -= n[1] / n2 * scale;
    grad[2] -= n[2] / n2 * scale;

    //== find closest point on boundary
    // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
    /* T3 m = diff(p1, p0); */
    /* T3 v = diff(p, p0); */
    /* T t = fmin(fmax(dot(m, v) / dot(m, m), 0.), 1.); */
    /* T d2 = len_squared(fma(v, -t, m)); */
    /* if (d2 < min_d2) */
    /*   closest_point = fma(p0, t, m); */
    /* min_d2 = fmin(min_d2, d2); */
  }

  /* T3 offset = diff(p, closest_point); */

  // blend solid angle gradient and offset based on distance to boundary
  /* T s = smoothstep(min_d2, 0, HARNACK_EPS); */
  /* s = 0; */
  /* grad[0] = (1 - s) * offset[0] + s * grad[0]; */
  /* grad[1] = (1 - s) * offset[1] + s * grad[1]; */
  /* grad[2] = (1 - s) * offset[2] + s * grad[2]; */

  T grad_norm = len(grad);
  grad[0] /= grad_norm;
  grad[1] /= grad_norm;
  grad[2] /= grad_norm;

  return make_float3(grad[0], grad[1], grad[2]);
}

ccl_device_inline float3
nonplanar_polygon_normal(KernelGlobals kg,
                         ccl_private ShaderData *sd,
                         ccl_private const Intersection *ccl_restrict isect,
                         ccl_private const Ray *ray)
{
  /* load vertices */
  const uint3 polygon_data = kernel_data_fetch(tri_vindex, isect->prim);
  uint polygon_start = polygon_data.x;
  uint polygon_size = polygon_data.y;
  const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

  float3 P = ray->P + isect->t * ray->D;
  float3 normal = ray_nonplanar_polygon_normal_T<double>(P, pts, polygon_size);

  // HACK: two-sided material
  /* if (dot(ray->D, normal) < 0) */
  /*   return -normal; */
  /* return normal; */

  /* return normal */
  if (object_negative_scale_applied(sd->object_flag)) {
    return -normal;
    /* return normalize(cross(v2 - v0, v1 - v0)); */
  }
  else {
    return normal;
    /* return normalize(cross(v1 - v0, v2 - v0)); */
  }
}

CCL_NAMESPACE_END
