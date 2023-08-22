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
template<typename T>
ccl_device bool ray_nonplanar_polygon_intersect_T(const float3 ray_Pf,
                                                  const float3 ray_Df,
                                                  const float ray_tmin,
                                                  const float ray_tmax,
                                                  const packed_float3 *pts,
                                                  const size_t N,
                                                  ccl_private float *isect_u,
                                                  ccl_private float *isect_v,
                                                  ccl_private float *isect_t)
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
  auto diff_f = [](const float3 &a, const T3 &b) -> T3 {
    return {(T)a.x - b[0], (T)a.y - b[1], (T)a.z - b[2]};
  };
  // a + s * b
  auto fma = [](const T3 &a, T s, const T3 &b) -> T3 {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
  };

  int max_iterations = 1500;
  T epsilon = 0.0001;
  T shift = 4. * M_PI;

  if (N < 3)
    return false;

  T3 ray_P = from_float3(ray_Pf);
  T3 ray_D = from_float3(ray_Df);
  T3 center = from_float3(pts[N]);

  // TODO: try doubles?
  // Fractional mod function that behaves the same as in GLSL
  auto glsl_mod = [](T x, T y) -> T { return x - y * std::floor(x / y); };

  auto close_to_zero = [&](T ang) -> bool {
    // Check if an angle is within epsilon of 0 (or 4π)
    T dis = fmin(ang, (T)(4 * M_PI) - ang);
    T tolScaling = 1;  // useGradStoppingCriterion ? Length(grad) : 1;
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

  auto solid_angle = [&](const T3 &x) -> T {
    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N + 1);
    std::vector<T> Lp;
    Lp.reserve(N + 1);
    for (int i = 0; i < N + 1; i++) {
      xp.push_back(diff_f(pts[i], x));
      Lp.push_back(len(xp[i]));
    }
    xp.push_back(diff(center, x));
    Lp.push_back(len(xp[N]));

    // Iterate over triangles used to triangulate the polygon
    std::complex<T> running_angle{1., 0.};
    for (int i = 0; i < N; i++) {
      int a = i;
      int b = (i + 1) % N;
      int c = N;

      // Add the solid angle of this triangle to the total
      std::complex<T> tri_angle{Lp[a] * Lp[b] * Lp[c] + dot(xp[a], xp[b]) * Lp[c] +
                                    dot(xp[b], xp[c]) * Lp[a] + dot(xp[a], xp[c]) * Lp[b],
                                dot(xp[a], cross(xp[b], xp[c]))};
      running_angle *= tri_angle;
    }
    return 2 * std::arg(running_angle);
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
    T val = glsl_mod((double)solid_angle(pos) - 2. * M_PI, 4. * M_PI);

    // Calculate the radius of a ball around the current point over which
    // we know the function is harmonic.  An easy way to identify such a
    // ball is to restrict to the sphere touching the closest point on the
    // polygon boundary.
    T3 closestPoint;
    T R = distance_to_boundary(pos, &closestPoint);

    // If we're close enough to the level set, or we've exceeded the
    // maximum number of iterations, assume there's a hit.
    if (close_to_zero(val) || R < epsilon || iter > max_iterations) {
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
  const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

  float u, v, t;
  if (ray_nonplanar_polygon_intersect_T<double>(P, dir, tmin, tmax, pts, N, &u, &v, &t)) {
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
// TODO: implement

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
  auto diff_f = [](const float3 &a, const T3 &b) -> T3 {
    return {(T)a.x - b[0], (T)a.y - b[1], (T)a.z - b[2]};
  };
  // a + s * b
  auto fma = [](const T3 &a, T s, const T3 &b) -> T3 {
    return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
  };

  T3 p = from_float3(pf);
  T3 grad{0, 0, 0};

  for (int i = 0; i < N; i++) {
    T3 p0 = from_float3(pts[(i + 0) % N]);
    T3 p1 = from_float3(pts[(i + 1) % N]);
    T3 g0 = diff(p0, p);
    T3 g1 = diff(p1, p);
    T3 n = cross(g1, g0);
    T n2 = len_squared(n);
    T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) + (-dot(g0, g1) + dot(g1, g1)) / len(g1));

    grad[0] -= n[0] / n2 * scale;
    grad[1] -= n[1] / n2 * scale;
    grad[2] -= n[2] / n2 * scale;
  }

  T grad_norm = len(grad);
  grad[0] /= grad_norm;
  grad[1] /= grad_norm;
  grad[2] /= grad_norm;

  return make_float3(grad[0], grad[1], grad[2]);
}

ccl_device_inline float3
nonplanar_polygon_normal(KernelGlobals kg,
                         ccl_private ShaderData *sd,
                         ccl_private const Intersection &ccl_restrict isect)
{
  /* load vertices */
  const uint3 polygon_data = kernel_data_fetch(tri_vindex, isect.prim);
  uint polygon_start = polygon_data.x;
  uint polygon_size = polygon_data.y;
  const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

  float3 normal = ray_nonplanar_polygon_normal_T<double>(sd->P, pts, polygon_size);

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
