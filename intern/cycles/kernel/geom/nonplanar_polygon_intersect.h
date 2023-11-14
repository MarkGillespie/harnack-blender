/* SPDX-FileCopyrightText: 2014-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Nonplanar polygon/ray intersections.
 *
 */

#pragma GCC diagnostic ignored "-Wdouble-promotion"

#pragma once

#include "kernel/sample/lcg.h"
#include "scene/nonplanar_polygon.h"  // scenario enums
#include "spherical_harmonics.h"
#include "util/debug.h"
#include "util/math_elliptic_integral.h"
#include <array>

CCL_NAMESPACE_BEGIN

// helpers
#include "vector_arithmetic.ipp"

// TODO: move back to math utils?
// int solid_angle_formula (from DNA_modifier_types.h):
//    MOD_HARNACK_TRIANGULATE = 0,
//    MOD_HARNACK_PREQUANTUM = 1,
//    MOD_HARNACK_GAUSS_BONNET = 2,

typedef struct spherical_harmonic_intersection_params {
  float3 ray_P;
  float3 ray_D;
  float ray_tmin;
  float ray_tmax;
  float R;
  uint l;
  int m;
  float epsilon;
  float levelset;
  float frequency;
  bool use_grad_termination;
  int max_iterations;
} spherical_harmonic_intersection_params;

template<typename T>
ccl_device bool ray_spherical_harmonic_intersect_T(
    const spherical_harmonic_intersection_params &params,
    ccl_private float *isect_u,
    ccl_private float *isect_v,
    ccl_private float *isect_t)
{
  using T3 = std::array<T, 3>;
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);

  T3 ray_P = from_float3(params.ray_P);
  T3 ray_D = from_float3(params.ray_D);

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  T radius = static_cast<T>(params.R);
  T outerRadius = static_cast<T>(1.25) * radius;  // TODO: make configurable?

  T unitBound = sphericalHarmonicBound(params.l, params.m);
  // since the spherical harmonic is homogeneous of degree l, we can bound it using a bound on the
  // unit sphere and our sphere radius
  T shift = unitBound * static_cast<T>(std::pow(outerRadius, params.l));

  // find the two times at which a ray intersects a sphere
  auto intersectSphere = [](const T3 &ro, const T3 &rd, T radius, T *t0, T *t1) -> bool {
    // solve | ro + t rd  |^2 == radius^2
    T a = dot(rd, rd);
    T b = static_cast<T>(2) * dot(rd, ro);
    T c = dot(ro, ro) - radius * radius;
    T discr = b * b - static_cast<T>(4) * a * c;
    if (discr < 0) {
      return false;
    }
    else {
      T sqrt_d = std::sqrt(discr);
      T q = (b > 0) ? static_cast<T>(-0.5) * (b + sqrt_d) : static_cast<T>(-0.5) * (b - sqrt_d);
      *t0 = q / a;
      *t1 = c / q;
      if (t1 < t0)
        std::swap(*t0, *t1);
      return true;
    }
  };

  auto distance_to_boundary = [&](const T3 &x) -> T { return outerRadius - len(x); };

  // Return the value of this harmonic polynomial at an evaluation point p,
  auto evaluatePolynomial = [&](const T3 &p) -> T {
    return evaluateSphericalHarmonic(params.l, params.m, p);
  };

  // TODO: take in analytic expressions?
  // finite difference gradient from https://iquilezles.org/articles/normalsSDF
  auto calculateGradient = [&](const T3 &p) -> T3 {
    const double eps = 0.005;
    T q0 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, -1, -1}));
    T q1 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, -1, 1}));
    T q2 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, 1, -1}));
    T q3 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, 1, 1}));
    return T3{static_cast<T>(.5773) * (q3 + q0 - q1 - q2),
              static_cast<T>(.5773) * (q3 - q0 - q1 + q2),
              static_cast<T>(.5773) * (q3 - q0 + q1 - q2)};
  };

  // Within the space of all values `levelset + 2π * frequency * k`, find the two
  // bracketing the current value of f lower_levelset is set to the smaller of the two,
  // and upper_levelset is set to the larger
  auto findEnclosingLevelsets =
      [](T f, T levelset, T frequency, T *lower_levelset, T *upper_levelset) {
        // 𝑓0 ← (𝑓(r𝑡)−𝜙)/(2𝜋𝜔)
        // 𝑓− ← (2𝜋𝜔)⌊𝑓0⌋ + 𝜙
        // 𝑓+ ← (2𝜋𝜔)⌈𝑓0⌉ + 𝜙
        T f0 = (f - levelset) / (static_cast<T>(2 * M_PI) * frequency);
        *lower_levelset = static_cast<T>(2 * M_PI) * frequency * std::floor(f0) + levelset;
        *upper_levelset = static_cast<T>(2 * M_PI) * frequency * std::ceil(f0) + levelset;
      };
  // Computes a conservative step size via Harnack bounds, using the value
  // fx at the current point, the radius R of a ball over which the function is
  // harmonic, the values lo_bound and up_bound of the closest level sets above/
  // below the current value, and a shift that makes the harmonic function
  // positive within this ball.
  auto getMaxStep = [](T fx, T R, T lo_bound, T up_bound, T shift) -> T {
    T w = (fx + shift) / (up_bound + shift);
    T v = (fx + shift) / (lo_bound + shift);
    T lo_r = -R / 2 * (v + 2 - std::sqrt(v * v + 8 * v));
    T up_r = R / 2 * (w + 2 - std::sqrt(w * w + 8 * w));

    return std::min(lo_r, up_r);
  };
  auto getMaxStep0 = [](T fx, T R, T bound, T shift) -> T {
    T a = (fx + shift) / (bound + shift);
    return R / 2 * std::abs(a + 2 - std::sqrt(a * a + 8 * a));
  };

  auto distanceToLevelset = [&](T f, T levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return std::abs(f - levelset) / scaling;
  };

  auto distanceToBothLevelsets =
      [&](T f, T lower_levelset, T upper_levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return fmin(f - lower_levelset, upper_levelset - f) / scaling;
  };

  // check if ray intersects sphere. If so, store two intersection times (t0 < t1)
  T t0, t1;
  bool hitSphere = intersectSphere(ray_P, ray_D, radius, &t0, &t1);

  // if we miss the sphere, there cannot be an intersection with the levelset within the sphere
  if (!hitSphere)
    return false;

  T t = fmax(t0, ray_tmin);     // start at first sphere intersection if it is ahead of us
  T tMax = fmin(ray_tmax, t1);  // only trace until second sphere intersection
  T ld = len(ray_D);

  // If we're in the periodic case, identify the two levelsets bracketing our starting position
  // Note that these should remain fixed as we step, since we never want to cross the levelsets
  T lower_levelset, upper_levelset;
  if (params.frequency > 0) {
    T3 pos = fma(ray_P, t, ray_D);
    findEnclosingLevelsets(
        evaluatePolynomial(pos), levelset, frequency, &lower_levelset, &upper_levelset);
  }

  int iter = 0;

  static bool exceeded_max = false;

  // Until we reach the maximum ray distance
  while (t < tMax) {
    // If we've exceeded the maximum number of iterations, print a warning
    if (iter >= params.max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf("Warning: exceeded maximum number of Harnack iterations.\n");
      }
    }

    T3 pos = fma(ray_P, t, ray_D);
    T f = evaluatePolynomial(pos);
    T3 grad;
    if (params.use_grad_termination)
      grad = calculateGradient(pos);

    // compute the distance to the level set.
    // if we're working periodically, this involves computing the closest levelsets
    // above and below us, which are also used later to compute a safe step size
    T dist;
    if (params.frequency < 0) {  // just looking for a single level set
      dist = distanceToLevelset(f, levelset, grad);
    }
    else {  // looking for periodic level sets
      dist = distanceToBothLevelsets(f, lower_levelset, upper_levelset, grad);
    }

    // If we're close enough to the level set, return a hit.
    if (dist < epsilon) {
      // TODO: add to normal code
      /*
          T3 grad = calculateGradient(pos);
          // shade points on the boundary sphere using the sphere's normal
          // ( for some reason, these points occasionally cause problems otherwise )
          if (1 - LengthSquared(Vector3d(pos)) < epsilon)
          grad = Vector3d(pos);
          return constructIntersection(t, pos, grad, rd, f);
      */
      *isect_t = t;
      *isect_u = f;
      *isect_v = ((T)iter) / ((T)params.max_iterations);
      return true;
    }

    T R = distance_to_boundary(pos);

    T step;
    if (params.frequency < 0) {  // just looking for a single level set
      step = getMaxStep0(f, R, levelset, shift);
    }
    else {  // looking for periodic level sets
      step = getMaxStep(f, R, lower_levelset, upper_levelset, shift);
    }

    t += step / ld;
    iter++;
  }

  return false;  // no intersection
}

typedef struct solid_angle_intersection_params {
  float3 ray_P;
  float3 ray_D;
  float ray_tmin;
  float ray_tmax;
  const packed_uint3 *loops;
  const packed_float3 *pts;
  uint n_loops;
  float epsilon;
  float levelset;
  float frequency;
  int solid_angle_formula;
  bool use_grad_termination;
  int max_iterations;
  bool clip_y;
} solid_angle_intersection_params;

static bool printed_disk_normal = false;
template<typename T>
ccl_device bool ray_nonplanar_polygon_intersect_T(const solid_angle_intersection_params &params,
                                                  ccl_private float *isect_u,
                                                  ccl_private float *isect_v,
                                                  ccl_private float *isect_t)
{
  using T3 = std::array<T, 3>;
  using T4 = std::array<T, 4>;
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);
  T shift = 4. * M_PI;

  T3 ray_P = from_float3(params.ray_P);
  T3 ray_D = from_float3(params.ray_D);
  uint globalStart = params.loops[0].x;

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  if (params.clip_y) {
    if (ray_P[1] <= 0 && ray_D[1] <= 0) {  // moving away from clipping plane
      return false;
    }
    else if (ray_P[1] <= 0) {  // moving towards clipping plane from far side
      ray_tmin = max(ray_tmin, -ray_P[1] / ray_D[1]);  // p + t * d = 0
    }
    else if (ray_D[1] <= 0) {  // moving towards clipping plane from near side
      ray_tmax = min(ray_tmax, -ray_P[1] / ray_D[1]);
    }
  }

  // classify each loop as a polygon or as an disk
  // HACK: count a polygon as a disk if it has five sides, and is disk-inscribed
  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;
  for (uint iL = 0; iL < params.n_loops; iL++) {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    //== Test if loop is "circular", i.e. 5-sided, equidistant from center point, and planar
    //     if so, store the center, normal, and radius
    T3 center, normal;
    T radius;
    bool isCircular = true;
    if (N != 5)
      isCircular = false;

    if (isCircular) {
      center = from_float3(params.pts[iStart + N]);

      // Store displacements from center
      std::vector<T3> rx;
      rx.reserve(N);
      for (int i = 0; i < N; i++)
        rx.push_back(diff_f(params.pts[iStart + i], center));

      // Test if points are equidistant from center, and store radius
      radius = len(rx[0]);
      for (int i = 1; i < N; i++) {
        T radius_i = len(rx[i]);
        if (std::abs(radius_i - radius) >= 0.01) {
          isCircular = false;
          break;
        }
      }

      if (isCircular) {
        // Test if points are coplanar (and store normal)
        normal = cross(rx[0], rx[1]);
        normalize(normal);
        for (int i = 1; i < N; i++) {
          int j = (i + 1) % N;
          T3 normal_i = cross(rx[i], rx[j]);

          // Check whether or not normal is parallel to normal_i
          if (std::abs(dot(normal, normal_i) - len(normal) * len(normal_i)) >= 0.01) {
            isCircular = false;
            break;
          }
        }
      }
    }

    if (isCircular) {
      // if we got here, the loop is circular. Store it as a disk
      diskCenters.push_back(center);
      diskNormals.push_back(normal);
      diskRadii.push_back(radius);

      if (!printed_disk_normal) {
        std::cout << ("normal: (" + std::to_string(normal[0]) + ", " + std::to_string(normal[1]) +
                      ", " + std::to_string(normal[2]) + ")")
                  << std::endl;
        printed_disk_normal = true;
      }
    }
    else {
      // if loop was not determined to be circular, record it as a polygon
      polygonLoops.push_back(iL);
    }
  }

  // Fractional mod function that behaves the same as in GLSL
  auto glsl_mod = [](T x, T y) -> T { return x - y * std::floor(x / y); };

  auto close_to_zero = [&](T ang, T lo_bound, T hi_bound, const T3 &grad) -> bool {
    // Check if an angle is within epsilon of 0 or 4π
    T dis = fmin(ang - lo_bound, hi_bound - ang);
    T tolScaling = params.use_grad_termination ? len(grad) : 1;
    return dis < epsilon * tolScaling;
  };

  auto squared_distance_to_polygon_boundary = [&](uint iL, const T3 &x, T3 *closest_point) -> T {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    // TODO maybe replace with numeric_limits<T>::infinity()?
    const T infinity = 100000.;
    T min_d2 = infinity;

    // compute closest distance to each polygon line segment
    for (int i = 0; i < N; i++) {
      T3 p1 = from_float3(params.pts[iStart + i]);
      T3 p2 = from_float3(params.pts[iStart + (i + 1) % N]);
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

    return min_d2;
  };

  auto squared_distance_to_disk_boundary = [&](uint iD, const T3 &x, T3 *closest_point) -> T {
    T3 displacement = diff(x, diskCenters[iD]);
    T L = dot(displacement, diskNormals[iD]);
    T r0 = len(fma(displacement, -L, diskNormals[iD]));
    T rm = diskRadii[iD];
    T d2 = std::pow(rm - r0, 2) + std::pow(L, 2);
    // if closest_point is not null, update it to track closest point
    if (false) {
      T3 rHat = fma(displacement, -L, diskNormals[iD]);
      *closest_point = fma(diskCenters[iD], rm / r0, rHat);
    }
    return d2;
  };

  auto distance_to_boundary = [&](const T3 &x, T3 *closest_point) -> T {
    const T infinity = 100000.;
    T min_d2 = infinity;
    // TODO: TKTKTKT closest point logic is broken
    for (uint iL : polygonLoops)
      min_d2 = fmin(min_d2, squared_distance_to_polygon_boundary(iL, x, closest_point));
    for (uint iD = 0; iD < diskCenters.size(); iD++)
      min_d2 = fmin(min_d2, squared_distance_to_disk_boundary(iD, x, closest_point));
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

  // returns solid angle of loop iL and adds gradient to grad
  auto triangulated_loop_solid_angle = [&](uint iL, const T3 &x, T3 &grad) -> T {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N + 1);
    std::vector<T> Lp;
    Lp.reserve(N + 1);
    for (int i = 0; i < N + 1; i++) {  // center = pts[N]
      xp.push_back(diff_f(params.pts[iStart + i], x));
      Lp.push_back(len(xp[i]));
    }

    // Iterate over triangles used to triangulate the polygon
    std::complex<T> running_angle{1., 0.};
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
      if (params.use_grad_termination) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        grad[0] += n[0] / n2 * scale;
        grad[1] += n[1] / n2 * scale;
        grad[2] += n[2] / n2 * scale;
      }
    }

    return 2 * std::arg(running_angle);
  };

  // sums solid angle of polygon loops and adds gradient to grad
  auto triangulated_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += triangulated_loop_solid_angle(iL, x, grad);
    return omega;
  };

  // returns solid angle of loop iL and adds gradient to grad
  auto prequantum_loop_solid_angle = [&](uint iL, const T3 &x, T3 &grad) -> T {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N);
    std::vector<T> Lp;
    Lp.reserve(N);
    for (int i = 0; i < N; i++) {
      xp.push_back(diff_f(params.pts[iStart + i], x));
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

      //== compute gradient
      if (params.use_grad_termination) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T3 n = cross(g0, g1);
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        grad[0] -= n[0] / n2 * scale;
        grad[1] -= n[1] / n2 * scale;
        grad[2] -= n[2] / n2 * scale;
      }
    }

    return static_cast<T>(-2.) * fiberArg(q0, qi);
  };

  // sums solid angle of polygon loops and adds gradient to grad
  auto prequantum_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += prequantum_loop_solid_angle(iL, x, grad);
    return omega;
  };

  // returns solid angle of loop iL and adds gradient to grad
  auto gauss_bonnet_loop_solid_angle = [&](uint iL, const T3 &x, T3 &grad) -> T {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N);
    std::vector<T> Lp;
    Lp.reserve(N);
    for (int i = 0; i < N; i++) {
      xp.push_back(diff_f(params.pts[iStart + i], x));
      Lp.push_back(len(xp[i]));
    }

    // Iterate over triangles used to triangulate the polygon
    T total_angle = 0.;
    for (int i = 0; i < N; i++) {
      int a = (i + N - 1) % N;
      int b = i;
      int c = (i + 1) % N;
      T3 n_prev = cross(xp[a], xp[b]);
      T3 n_next = cross(xp[b], xp[c]);

      total_angle += atan2(dot(xp[b], cross(n_prev, n_next)) / Lp[b], dot(n_prev, n_next));

      //== compute gradient
      if (params.use_grad_termination) {
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
    return static_cast<T>(2. * M_PI) - total_angle;
  };

  // sums solid angle of polygon loops and adds gradient to grad
  auto gauss_bonnet_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += gauss_bonnet_loop_solid_angle(iL, x, grad);
    return omega;
  };

  // returns solid angle of disk iD and adds gradient to grad
  auto disk_solid_angle = [&](uint iD, const T3 &x, T3 &grad) -> T {
    //     Input, double ERRTOL, the error tolerance.
    //     Relative error due to truncation is less than
    //       ERRTOL ^ 6 / (4 * (1 - ERRTOL)).
    //     Sample choices:
    //       ERRTOL   Relative truncation error less than
    //       1.D-3    3.D-19
    //       3.D-3    2.D-16
    //       1.D-2    3.D-13
    //       3.D-2    2.D-10
    //       1.D-1    3.D-7
    // double elliptic_errtol = 0.75;
    double elliptic_errtol = 0.01;

    // Formulas from Solid Angle Calculation for a Circular Disk by Paxton 1959
    T3 displacement = diff(x, diskCenters[iD]);  // might need to switch to diff_f
    const T3 &zHat = diskNormals[iD];
    T L = dot(displacement, diskNormals[iD]);
    T3 inPlane = fma(displacement, -L, zHat);
    T r0 = len(inPlane);
    T sign = copysign(1, L);
    T aL = std::abs(L);  // use absolute value of length when evaluating solid angle to shift
                         // branch cut to the horizontal plane
    T rm = diskRadii[iD];
    T alphaSq = 4. * r0 * rm / std::pow(r0 + rm, 2);
    T R1Sq = std::pow(L, 2) + std::pow(r0 - rm, 2);
    T Rmax = std::sqrt(std::pow(L, 2) + std::pow(r0 + rm, 2));
    T kSq = 1 - R1Sq / (Rmax * Rmax);
    T3 rHat = over(inPlane, r0);

    T F = elliptic_fm(kSq, elliptic_errtol);

    // TODO: adjust tolerance
    // ( When r0 and rm are too close, alphaSq goes to 1, which leads to a singularity in
    // elliptic_pim )
    T omega;
    if (std::abs(r0 - rm) < 1e-3) {
      omega = sign * (M_PI - 2 * aL / Rmax * F);
    }
    else if (r0 < rm) {
      omega = sign * (2 * M_PI - 2 * aL / Rmax * F +
                      2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                          elliptic_pim(alphaSq, kSq, elliptic_errtol));
    }
    else if (r0 > rm) {
      omega = sign * (-2 * aL / Rmax * F + 2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                                               elliptic_pim(alphaSq, kSq, elliptic_errtol));
    }

    if (true) {
      T E = elliptic_em(kSq, elliptic_errtol);
      T Br = -2 * aL / (r0 * Rmax) * (-F + (rm * rm + r0 * r0 + aL * aL) / R1Sq * E);
      T Bz = -2 / (Rmax) * (F + (rm * rm - r0 * r0 - aL * aL) / R1Sq * E);

      for (size_t i = 0; i < 3; i++)
        (grad)[i] += sign * Br * rHat[i] + Bz * zHat[i];
    }
    return omega;
  };

  // sums solid angle of polygon loops and disks and adds gradient to grad
  auto total_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = params.solid_angle_formula == 0 ? triangulated_solid_angle(x, grad) :
              params.solid_angle_formula == 1 ? prequantum_solid_angle(x, grad) :
                                                gauss_bonnet_solid_angle(x, grad);

    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(iD, x, grad);

    return omega;
  };

  /* Find intersection with Harnack tracing */

  T t = ray_tmin;
  int iter = 0;
  T lo_bound = 0;
  T hi_bound = 4. * M_PI;

  T ld = len(ray_D);

  static bool exceeded_max = false;
  while (t < ray_tmax) {
    T3 pos = fma(ray_P, t, ray_D);

    // If we've exceeded the maximum number of iterations,
    // print a warning
    if (iter > params.max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf("Warning: exceeded maximum number of Harnack iterations.\n");
      }

      return false;
    }

    T3 grad{0, 0, 0};
    T omega = total_solid_angle(pos, grad);

    // To get the most aggressive Harnack bound, we first find a
    // representative of the solid angle, shifted by the target level set
    // value, within the range [0,4π).  Only then do we apply the shift.
    T val = glsl_mod(omega - levelset, static_cast<T>(4. * M_PI));

    if (frequency > 0) {
      lo_bound = 0;
      hi_bound = 4. * M_PI;
      // add four additional level sets
      if (val < frequency * hi_bound) {
        hi_bound = frequency * hi_bound;
      }
      else if (val < 2 * frequency * hi_bound) {
        lo_bound = frequency * hi_bound;
        hi_bound = 2 * frequency * hi_bound;
      }
      else if (val < hi_bound - 2 * frequency * hi_bound) {
        lo_bound = 2 * frequency * hi_bound;
        hi_bound = hi_bound - 2 * frequency * hi_bound;
      }
      else if (val < hi_bound - frequency * hi_bound) {
        lo_bound = hi_bound - 2 * frequency * hi_bound;
        hi_bound = hi_bound - frequency * hi_bound;
      }
      else {
        lo_bound = hi_bound - frequency * hi_bound;
      }
    }

    // Calculate the radius of a ball around the current point over which
    // we know the function is harmonic.  An easy way to identify such a
    // ball is to restrict to the sphere touching the closest point on the
    // polygon boundary.
    T3 closestPoint;
    T R = distance_to_boundary(pos, &closestPoint);

    // If we're close enough to the level set, or we've exceeded the
    // maximum number of iterations, assume there's a hit.
    if (close_to_zero(val, lo_bound, hi_bound, grad) || R < epsilon ||
        iter > params.max_iterations)
    {
      // if (R < epsilon)   grad = pos - closestPoint; // TODO: this?
      *isect_t = t;
      *isect_u = omega / static_cast<T>(4. * M_PI);
      *isect_v = ((T)iter) / ((T)params.max_iterations);
      return true;
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

  // std::cout << (std::string("UP HERE ")) << std::endl;
  const packed_uint3 *polygon_data = &kernel_data_fetch(tri_vindex, prim);
  uint polygon_start = polygon_data[0].x;
  uint N = polygon_data[0].y;
  uint scenario = polygon_data[0].z;

  float u, v, t;
  bool found_intersection = false;
  switch (scenario) {
    case MOD_HARNACK_NONPLANAR_POLYGON: {
      solid_angle_intersection_params sa_params;
      sa_params.ray_P = P;
      sa_params.ray_D = dir;
      sa_params.ray_tmin = tmin;
      sa_params.ray_tmax = tmax;

      sa_params.loops = polygon_data;
      sa_params.pts = &kernel_data_fetch(tri_verts, polygon_start);
      float3 params = sa_params.pts[N + 1];
      sa_params.epsilon = params.x;
      sa_params.levelset = params.y * ((float)(4. * M_PI));
      uint properties = static_cast<uint>(params.z);
      sa_params.max_iterations = properties >> 6;
      sa_params.solid_angle_formula = (properties >> 2) & 0xf;
      uint precision = (properties >> 1) & 1;
      sa_params.use_grad_termination = properties & 1;
      params = sa_params.pts[N + 2];
      sa_params.frequency = params.x;
      sa_params.clip_y = static_cast<bool>(params.y);
      sa_params.n_loops = static_cast<uint>(params.z);

      if (precision == 0) {
        found_intersection = ray_nonplanar_polygon_intersect_T<float>(sa_params, &u, &v, &t);
      }
      else if (precision == 1) {
        found_intersection = ray_nonplanar_polygon_intersect_T<double>(sa_params, &u, &v, &t);
      }
      break;
    }
    case MOD_HARNACK_DISK_SHELL: {  // disk shell
      // TODO
      break;
    }
    case MOD_HARNACK_SPHERICAL_HARMONIC: {  // spherical harmonic
      spherical_harmonic_intersection_params sh_params;
      sh_params.ray_P = P;
      sh_params.ray_D = dir;
      sh_params.ray_tmin = tmin;
      sh_params.ray_tmax = tmax;

      const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

      sh_params.R = pts[0].x;
      sh_params.l = static_cast<uint>(pts[0].y);
      sh_params.m = static_cast<int>(pts[0].z);

      float3 params = pts[1];
      sh_params.epsilon = params.x;
      sh_params.levelset = params.y;
      uint properties = static_cast<uint>(params.z);
      sh_params.max_iterations = properties >> 6;
      uint precision = (properties >> 1) & 1;

      sh_params.frequency = pts[2].x;

      if (precision == 0) {
        found_intersection = ray_spherical_harmonic_intersect_T<float>(sh_params, &u, &v, &t);
      }
      else if (precision == 1) {
        found_intersection = ray_spherical_harmonic_intersect_T<double>(sh_params, &u, &v, &t);
      }

      // std::cout << (std::string("HERE ") + (found_intersection ? "HIT" : "MISS")) << std::endl;

      break;
    }
    case MOD_HARNACK_RIEMANN_SURFACE: {  // Riemann surface
      // TODO
      break;
    }
  }
  if (found_intersection) {
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
                                                 const packed_uint3 *loops,
                                                 const packed_float3 *pts,
                                                 const uint n_loops)
{
  using T3 = std::array<T, 3>;
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };
  /* auto smoothstep = [](T x, T start, T end) -> T {  // https://docs.gl/sl4/smoothstep */
  /*   // normalize and clamp to [0, 1] */
  /*   x = fmin(fmax(((x - start) / (end - start)), 0), 1); */

  /*   return x * x * (3. - 2. * x); */
  /* }; */

  // find solid angle gradient and closest point on boundary curve
  uint globalStart = loops[0].x;
  T3 p = from_float3(pf);

  // classify each loop as a polygon or as an disk
  // HACK: count a polygon as a disk if it has five sides, and is disk-inscribed
  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;
  for (uint iL = 0; iL < n_loops; iL++) {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

    //== Test if loop is "circular", i.e. 5-sided, equidistant from center point, and planar
    //     if so, store the center, normal, and radius
    T3 center, normal;
    T radius;
    bool isCircular = true;
    if (N != 5)
      isCircular = false;

    if (isCircular) {
      center = from_float3(pts[iStart + N]);

      // Store displacements from center
      std::vector<T3> rx;
      rx.reserve(N);
      for (int i = 0; i < N; i++)
        rx.push_back(diff_f(pts[iStart + i], center));

      // Test if points are equidistant from center, and store radius
      radius = len(rx[0]);
      for (int i = 1; i < N; i++) {
        T radius_i = len(rx[i]);
        if (std::abs(radius_i - radius) >= 0.01) {
          isCircular = false;
          break;
        }
      }

      if (isCircular) {
        // Test if points are coplanar (and store normal)
        normal = cross(rx[0], rx[1]);
        normalize(normal);
        for (int i = 1; i < N; i++) {
          int j = (i + 1) % N;
          T3 normal_i = cross(rx[i], rx[j]);

          // Check whether or not normal is parallel to normal_i
          if (std::abs(dot(normal, normal_i) - len(normal) * len(normal_i)) >= 0.01) {
            isCircular = false;
            break;
          }
        }
      }
    }

    if (isCircular) {
      // if we got here, the loop is circular. Store it as a disk
      diskCenters.push_back(center);
      diskNormals.push_back(normal);
      diskRadii.push_back(radius);
    }
    else {
      // if loop was not determined to be circular, record it as a polygon
      polygonLoops.push_back(iL);
    }
  }

  // returns solid angle of loop iL and adds gradient to grad
  auto triangulated_loop_solid_angle = [&](uint iL, const T3 &x, T3 *grad) -> T {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N + 1);
    std::vector<T> Lp;
    Lp.reserve(N + 1);
    for (int i = 0; i < N + 1; i++) {  // center = pts[N]
      xp.push_back(diff_f(pts[iStart + i], x));
      Lp.push_back(len(xp[i]));
    }

    // Iterate over triangles used to triangulate the polygon
    std::complex<T> running_angle{1., 0.};
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
      if (grad) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
    }

    return 2 * std::arg(running_angle);
  };

  // returns solid angle of disk iD and adds gradient to grad
  auto disk_solid_angle = [&](uint iD, const T3 &x, T3 *grad) -> T {
    //     Input, double ERRTOL, the error tolerance.
    //     Relative error due to truncation is less than
    //       ERRTOL ^ 6 / (4 * (1 - ERRTOL)).
    //     Sample choices:
    //       ERRTOL   Relative truncation error less than
    //       1.D-3    3.D-19
    //       3.D-3    2.D-16
    //       1.D-2    3.D-13
    //       3.D-2    2.D-10
    //       1.D-1    3.D-7
    // double elliptic_errtol = 0.75;
    double elliptic_errtol = 0.01;

    // Formulas from Solid Angle Calculation for a Circular Disk by Paxton 1959
    T3 displacement = diff(x, diskCenters[iD]);  // might need to switch to diff_f
    const T3 &zHat = diskNormals[iD];
    T L = dot(displacement, diskNormals[iD]);
    T3 inPlane = fma(displacement, -L, zHat);
    T r0 = len(inPlane);
    T sign = copysign(1, L);
    T aL = std::abs(L);  // use absolute value of length when evaluating solid angle to shift
                         // branch cut to the horizontal plane
    T rm = diskRadii[iD];
    T alphaSq = 4. * r0 * rm / std::pow(r0 + rm, 2);
    T R1Sq = std::pow(L, 2) + std::pow(r0 - rm, 2);
    T Rmax = std::sqrt(std::pow(L, 2) + std::pow(r0 + rm, 2));
    T kSq = 1 - R1Sq / (Rmax * Rmax);
    T3 rHat = over(inPlane, r0);

    T F = elliptic_fm(kSq, elliptic_errtol);

    // TODO: adjust tolerance
    // ( When r0 and rm are too close, alphaSq goes to 1, which leads to a singularity in
    // elliptic_pim )
    T omega;
    if (std::abs(r0 - rm) < 1e-3) {
      omega = sign * (M_PI - 2 * aL / Rmax * F);
    }
    else if (r0 < rm) {
      omega = sign * (2 * M_PI - 2 * aL / Rmax * F +
                      2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                          elliptic_pim(alphaSq, kSq, elliptic_errtol));
    }
    else if (r0 > rm) {
      omega = sign * (-2 * aL / Rmax * F + 2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                                               elliptic_pim(alphaSq, kSq, elliptic_errtol));
    }

    if (grad) {
      T E = elliptic_em(kSq, elliptic_errtol);
      T Br = -2 * aL / (r0 * Rmax) * (-F + (rm * rm + r0 * r0 + aL * aL) / R1Sq * E);
      T Bz = -2 / (Rmax) * (F + (rm * rm - r0 * r0 - aL * aL) / R1Sq * E);

      for (size_t i = 0; i < 3; i++)
        (*grad)[i] += sign * Br * rHat[i] + Bz * zHat[i];
    }
    return omega;
  };

  auto triangulated_solid_angle = [&](const T3 &x, T3 *grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += triangulated_loop_solid_angle(iL, x, grad);
    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(iD, x, grad);
    return omega;
  };

  T3 grad{0, 0, 0};
  if (false) {
    T h = 0.000001;
    T omega = triangulated_solid_angle(p, nullptr);
    T omega_x = triangulated_solid_angle(T3{p[0] + h, p[1], p[2]}, nullptr);
    T omega_y = triangulated_solid_angle(T3{p[0], p[1] + h, p[2]}, nullptr);
    T omega_z = triangulated_solid_angle(T3{p[0], p[1], p[2] + h}, nullptr);

    T fd_df_dx = (omega_x - omega) / h;
    T fd_df_dy = (omega_y - omega) / h;
    T fd_df_dz = (omega_z - omega) / h;

    grad = T3{fd_df_dx, fd_df_dy, fd_df_dz};
  }

  triangulated_solid_angle(p, &grad);

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

template<typename T>
ccl_device float3 ray_spherical_harmonic_normal_T(const float3 pf, uint m, int l)
{
  using T3 = std::array<T, 3>;
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };

  T3 p = from_float3(pf);

  // Return the value of this harmonic polynomial at an evaluation point p,
  auto evaluatePolynomial = [&](const T3 &p) -> T { return evaluateSphericalHarmonic(l, m, p); };

  // TODO: take in analytic expressions?
  // finite difference gradient from https://iquilezles.org/articles/normalsSDF
  auto calculateGradient = [&](const T3 &p) -> T3 {
    const double eps = 0.05;
    T q0 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, -1, -1}));
    T q1 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, -1, 1}));
    T q2 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, 1, -1}));
    T q3 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, 1, 1}));
    return T3{static_cast<T>(.5773) * (q3 + q0 - q1 - q2),
              static_cast<T>(.5773) * (q3 - q0 - q1 + q2),
              static_cast<T>(.5773) * (q3 - q0 + q1 - q2)};
  };

  T3 grad = calculateGradient(p);
  normalize(grad);

  return make_float3(grad[0], grad[1], grad[2]);
}

ccl_device_inline float3
nonplanar_polygon_normal(KernelGlobals kg,
                         ccl_private ShaderData *sd,
                         ccl_private const Intersection *ccl_restrict isect,
                         ccl_private const Ray *ray)
{
  // load data
  const packed_uint3 *polygon_data = &kernel_data_fetch(tri_vindex, isect->prim);
  uint polygon_start = polygon_data[0].x;
  uint N = polygon_data[0].y;
  uint scenario = polygon_data[0].z;

  float3 P = ray->P + isect->t * ray->D;
  float3 normal;

  switch (scenario) {
    case MOD_HARNACK_NONPLANAR_POLYGON: {
      const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);
      uint n_loops = static_cast<uint>(pts[N + 2].z);
      normal = ray_nonplanar_polygon_normal_T<double>(P, polygon_data, pts, n_loops);
      break;
    }
    case MOD_HARNACK_DISK_SHELL: {  // disk shell
      // TODO
      break;
    }
    case MOD_HARNACK_SPHERICAL_HARMONIC: {  // spherical harmonic
      const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

      float R = pts[0].x;
      // special case for points on sphere
      if (fabsf(P.x * P.x + P.y * P.y + P.z * P.z - R * R) < (float)1e-5) {
        normal = normalize(P);
      }
      else {
        uint l = static_cast<uint>(pts[0].y);
        int m = static_cast<int>(pts[0].z);
        normal = ray_spherical_harmonic_normal_T<double>(P, m, l);
      }
      break;
    }
    case MOD_HARNACK_RIEMANN_SURFACE: {  // Riemann surface
      // TODO
      break;
    }
  }

  /* return normal */
  if (object_negative_scale_applied(sd->object_flag)) {
    return -normal;
  }
  else {
    return normal;
  }
}

CCL_NAMESPACE_END
