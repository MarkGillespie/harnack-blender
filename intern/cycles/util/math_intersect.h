/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef __UTIL_MATH_INTERSECT_H__
#define __UTIL_MATH_INTERSECT_H__

#include <complex>
#include <iostream>
#include <vector>

CCL_NAMESPACE_BEGIN

/* Ray Intersection */

ccl_device bool ray_sphere_intersect(float3 ray_P,
                                     float3 ray_D,
                                     float ray_tmin,
                                     float ray_tmax,
                                     float3 sphere_P,
                                     float sphere_radius,
                                     ccl_private float3 *isect_P,
                                     ccl_private float *isect_t)
{
  const float3 d_vec = sphere_P - ray_P;
  const float r_sq = sphere_radius * sphere_radius;
  const float d_sq = dot(d_vec, d_vec);
  const float d_cos_theta = dot(d_vec, ray_D);

  if (d_sq > r_sq && d_cos_theta < 0.0f) {
    /* Ray origin outside sphere and points away from sphere. */
    return false;
  }

  const float d_sin_theta_sq = d_sq - d_cos_theta * d_cos_theta;

  if (d_sin_theta_sq > r_sq) {
    /* Closest point on ray outside sphere. */
    return false;
  }

  /* Law of cosines. */
  const float t = d_cos_theta - copysignf(sqrtf(r_sq - d_sin_theta_sq), d_sq - r_sq);

  if (t > ray_tmin && t < ray_tmax) {
    *isect_t = t;
    *isect_P = ray_P + ray_D * t;
    return true;
  }

  return false;
}

ccl_device bool ray_aligned_disk_intersect(float3 ray_P,
                                           float3 ray_D,
                                           float ray_tmin,
                                           float ray_tmax,
                                           float3 disk_P,
                                           float disk_radius,
                                           ccl_private float3 *isect_P,
                                           ccl_private float *isect_t)
{
  /* Aligned disk normal. */
  float disk_t;
  const float3 disk_N = normalize_len(ray_P - disk_P, &disk_t);
  const float div = dot(ray_D, disk_N);
  if (UNLIKELY(div == 0.0f)) {
    return false;
  }
  /* Compute t to intersection point. */
  const float t = -disk_t / div;
  if (!(t > ray_tmin && t < ray_tmax)) {
    return false;
  }
  /* Test if within radius. */
  float3 P = ray_P + ray_D * t;
  if (len_squared(P - disk_P) > disk_radius * disk_radius) {
    return false;
  }
  *isect_P = P;
  *isect_t = t;
  return true;
}

ccl_device bool ray_disk_intersect(float3 ray_P,
                                   float3 ray_D,
                                   float ray_tmin,
                                   float ray_tmax,
                                   float3 disk_P,
                                   float3 disk_N,
                                   float disk_radius,
                                   ccl_private float3 *isect_P,
                                   ccl_private float *isect_t)
{
  const float3 vp = ray_P - disk_P;
  const float dp = dot(vp, disk_N);
  const float cos_angle = dot(disk_N, -ray_D);
  if (dp * cos_angle > 0.f)  // front of light
  {
    float t = dp / cos_angle;
    if (t < 0.f) { /* Ray points away from the light. */
      return false;
    }
    float3 P = ray_P + t * ray_D;
    float3 T = P - disk_P;

    if (dot(T, T) < sqr(disk_radius) && (t > ray_tmin && t < ray_tmax)) {
      *isect_P = ray_P + t * ray_D;
      *isect_t = t;
      return true;
    }
  }
  return false;
}

/* Custom rcp, cross and dot implementations that match Embree bit for bit. */
ccl_device_forceinline float ray_triangle_rcp(const float x)
{
#ifdef __KERNEL_NEON__
  /* Move scalar to vector register and do rcp. */
  __m128 a;
  a[0] = x;
  float32x4_t reciprocal = vrecpeq_f32(a);
  reciprocal = vmulq_f32(vrecpsq_f32(a, reciprocal), reciprocal);
  reciprocal = vmulq_f32(vrecpsq_f32(a, reciprocal), reciprocal);
  return reciprocal[0];
#elif defined(__KERNEL_SSE__)
  const __m128 a = _mm_set_ss(x);
  const __m128 r = _mm_rcp_ss(a);

#  ifdef __KERNEL_AVX2_
  return _mm_cvtss_f32(_mm_mul_ss(r, _mm_fnmadd_ss(r, a, _mm_set_ss(2.0f))));
#  else
  return _mm_cvtss_f32(_mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(r, a))));
#  endif
#else
  return 1.0f / x;
#endif
}

ccl_device_inline float ray_triangle_dot(const float3 a, const float3 b)
{
#if defined(__KERNEL_SSE41__) && defined(__KERNEL_SSE__)
  return madd(make_float4(a.x),
              make_float4(b.x),
              madd(make_float4(a.y), make_float4(b.y), make_float4(a.z) * make_float4(b.z)))[0];
#else
  return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

ccl_device_inline float3 ray_triangle_cross(const float3 a, const float3 b)
{
#if defined(__KERNEL_SSE41__) && defined(__KERNEL_SSE__)
  return make_float3(
      msub(make_float4(a.y), make_float4(b.z), make_float4(a.z) * make_float4(b.y))[0],
      msub(make_float4(a.z), make_float4(b.x), make_float4(a.x) * make_float4(b.z))[0],
      msub(make_float4(a.x), make_float4(b.y), make_float4(a.y) * make_float4(b.x))[0]);
#else
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
#endif
}

ccl_device_forceinline bool ray_triangle_intersect(const float3 ray_P,
                                                   const float3 ray_D,
                                                   const float ray_tmin,
                                                   const float ray_tmax,
                                                   const float3 tri_a,
                                                   const float3 tri_b,
                                                   const float3 tri_c,
                                                   ccl_private float *isect_u,
                                                   ccl_private float *isect_v,
                                                   ccl_private float *isect_t)
{
  /* This implementation matches the Plücker coordinates triangle intersection
   * in Embree. */

  /* Calculate vertices relative to ray origin. */
  const float3 v0 = tri_a - ray_P;
  const float3 v1 = tri_b - ray_P;
  const float3 v2 = tri_c - ray_P;

  /* Calculate triangle edges. */
  const float3 e0 = v2 - v0;
  const float3 e1 = v0 - v1;
  const float3 e2 = v1 - v2;

  /* Perform edge tests. */
  const float U = ray_triangle_dot(ray_triangle_cross(e0, v2 + v0), ray_D);
  const float V = ray_triangle_dot(ray_triangle_cross(e1, v0 + v1), ray_D);
  const float W = ray_triangle_dot(ray_triangle_cross(e2, v1 + v2), ray_D);

  const float UVW = U + V + W;
  const float eps = FLT_EPSILON * fabsf(UVW);
  const float minUVW = min(U, min(V, W));
  const float maxUVW = max(U, max(V, W));

  if (!(minUVW >= -eps || maxUVW <= eps)) {
    return false;
  }

  /* Calculate geometry normal and denominator. */
  const float3 Ng1 = ray_triangle_cross(e1, e0);
  const float3 Ng = Ng1 + Ng1;
  const float den = dot(Ng, ray_D);
  /* Avoid division by 0. */
  if (UNLIKELY(den == 0.0f)) {
    return false;
  }

  /* Perform depth test. */
  const float T = dot(v0, Ng);
  const float t = T / den;
  if (!(t >= ray_tmin && t <= ray_tmax)) {
    return false;
  }

  const float rcp_uvw = (fabsf(UVW) < 1e-18f) ? 0.0f : ray_triangle_rcp(UVW);
  *isect_u = min(U * rcp_uvw, 1.0f);
  *isect_v = min(V * rcp_uvw, 1.0f);
  *isect_t = t;
  return true;
}

ccl_device_forceinline bool ray_triangle_intersect_self(const float3 ray_P,
                                                        const float3 ray_D,
                                                        const float3 verts[3])
{
  /* Matches logic in ray_triangle_intersect, self intersection test to validate
   * if a ray is going to hit self or might incorrectly hit a neighboring triangle. */

  /* Calculate vertices relative to ray origin. */
  const float3 v0 = verts[0] - ray_P;
  const float3 v1 = verts[1] - ray_P;
  const float3 v2 = verts[2] - ray_P;

  /* Calculate triangle edges. */
  const float3 e0 = v2 - v0;
  const float3 e1 = v0 - v1;
  const float3 e2 = v1 - v2;

  /* Perform edge tests. */
  const float U = ray_triangle_dot(ray_triangle_cross(v2 + v0, e0), ray_D);
  const float V = ray_triangle_dot(ray_triangle_cross(v0 + v1, e1), ray_D);
  const float W = ray_triangle_dot(ray_triangle_cross(v1 + v2, e2), ray_D);

  const float eps = FLT_EPSILON * fabsf(U + V + W);
  const float minUVW = min(U, min(V, W));
  const float maxUVW = max(U, max(V, W));

  /* Note the extended epsilon compared to ray_triangle_intersect, to account
   * for intersections with neighboring triangles that have an epsilon. */
  return (minUVW >= eps || maxUVW <= -eps);
}

/* Tests for an intersection between a ray and a quad defined by
 * its midpoint, normal and sides.
 * If ellipse is true, hits outside the ellipse that's enclosed by the
 * quad are rejected.
 */
ccl_device bool ray_quad_intersect(float3 ray_P,
                                   float3 ray_D,
                                   float ray_tmin,
                                   float ray_tmax,
                                   float3 quad_P,
                                   float3 inv_quad_u,
                                   float3 inv_quad_v,
                                   float3 quad_n,
                                   ccl_private float3 *isect_P,
                                   ccl_private float *isect_t,
                                   ccl_private float *isect_u,
                                   ccl_private float *isect_v,
                                   bool ellipse)
{
  /* Perform intersection test. */
  float t = -(dot(ray_P, quad_n) - dot(quad_P, quad_n)) / dot(ray_D, quad_n);
  if (!(t > ray_tmin && t < ray_tmax)) {
    return false;
  }
  const float3 hit = ray_P + t * ray_D;
  const float3 inplane = hit - quad_P;
  const float u = dot(inplane, inv_quad_u);
  if (u < -0.5f || u > 0.5f) {
    return false;
  }
  const float v = dot(inplane, inv_quad_v);
  if (v < -0.5f || v > 0.5f) {
    return false;
  }
  if (ellipse && (u * u + v * v > 0.25f)) {
    return false;
  }
  /* Store the result. */
  /* TODO(sergey): Check whether we can avoid some checks here. */
  if (isect_P != NULL)
    *isect_P = hit;
  if (isect_t != NULL)
    *isect_t = t;

  /* NOTE: Return barycentric coordinates in the same notation as Embree and OptiX. */
  if (isect_u != NULL)
    *isect_u = v + 0.5f;
  if (isect_v != NULL)
    *isect_v = -u - v;

  return true;
}

/* Tests for an intersection between a ray and a nonplanar polygon defined as
 * the 2 PI levelset of the solid angle function.
 * Comptutes the intersection via Harnack tracing
 */
ccl_device bool ray_nonplanar_polygon_intersect(const float3 ray_P,
                                                const float3 ray_D,
                                                const float ray_tmin,
                                                const float ray_tmax,
                                                const packed_float3 *tri_pts,
                                                const size_t N,
                                                ccl_private float *isect_u,
                                                ccl_private float *isect_v,
                                                ccl_private float *isect_t)
{
  int max_iterations = 1500;
  float epsilon = 0.0001f;
  float shift = static_cast<float>(4. * M_PI);

  if (N < 3)
    return false;

  /* std::string info = "intersecting: " + std::to_string(N); */
  /* for (size_t i = 0; i < N; i++) { */
  /*   info += " | " + std::to_string(tri_pts[i].x) + " " + std::to_string(tri_pts[i].y) + " " + */
  /*           std::to_string(tri_pts[i].z); */
  /* } */
  /* info += "\n"; */
  /* std::cout << info; */

  /* return false; */

  float3 center = zero_float3();
  for (size_t i = 0; i < N; i++)
    center += tri_pts[i];
  center /= N;

  // TODO: try doubles?
  auto close_to_zero = [&](float ang) -> bool {
    // Check if an angle is within epsilon of 0 (or 4π)
    float dis = fmin(ang, (float)(4 * M_PI) - ang);
    float tolScaling = 1;  // useGradStoppingCriterion ? Length(grad) : 1;
    return dis < epsilon * tolScaling;
  };

  auto distance_to_boundary = [&](const float3 &x, float3 *closest_point) -> float {
    // TODO maybe replace with numeric_limits<float>::infinity()?
    const float infinity = 100000.f;
    float min_d2 = infinity;

    // compute closest distance to each polygon line segment
    for (int i = 0; i < N; i++) {  // set up loop bounds so that it doesn't run if pts is empty
      float3 p1 = tri_pts[i];
      float3 p2 = tri_pts[(i + 1) % N];
      float3 m = p2 - p1;
      float3 v = x - p1;
      // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
      float t = fmin(fmax(dot(m, v) / dot(m, m), 0.f), 1.f);
      float d2 = len_squared(v - t * m);
      // if closestPoint is not null, update it to track closest point
      if (closest_point && d2 < min_d2)
        *closest_point = p1 + t * m;
      min_d2 = fmin(min_d2, d2);
    }

    return std::sqrt(min_d2);
  };

  // Computes a conservative step size via Harnack bounds, using the value
  // fx at the current point, the radius R of a ball over which the function is
  // harmonic, the values lo_bound and up_bound of the closest level sets above/
  // below the current value, and a shift that makes the harmonic function
  // positive within this ball.
  auto get_max_step =
      [&](float fx, float R, float lo_bound, float up_bound, float shift) -> float {
    float w = (fx + shift) / (up_bound + shift);
    float v = (fx + shift) / (lo_bound + shift);
    float lo_r = -R / 2 * (v + 2 - std::sqrt(v * v + 8 * v));
    float up_r = R / 2 * (w + 2 - std::sqrt(w * w + 8 * w));

    return std::min(lo_r, up_r);
  };

  auto solid_angle = [&](const float3 &x) -> float {
    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<float3> xp;
    xp.reserve(N + 1);
    std::vector<float> Lp;
    Lp.reserve(N + 1);
    for (int i = 0; i < N + 1; i++) {
      xp.push_back(tri_pts[i] - x);
      Lp.push_back(len(xp[i]));
    }
    xp.push_back(center - x);
    Lp.push_back(len(xp[N]));

    // Iterate over triangles used to triangulate the polygon
    std::complex<float> running_angle{1., 0.};
    for (int i = 0; i < N; i++) {
      int a = i;
      int b = (i + 1) % N;
      int c = N;

      // Add the solid angle of this triangle to the total
      std::complex<float> tri_angle{Lp[a] * Lp[b] * Lp[c] + dot(xp[a], xp[b]) * Lp[c] +
                                        dot(xp[b], xp[c]) * Lp[a] + dot(xp[a], xp[c]) * Lp[b],
                                    dot(xp[a], cross(xp[b], xp[c]))};
      running_angle *= tri_angle;
    }
    return 2 * std::arg(running_angle);
  };

  /* Find intersection with Harnack tracing */

  float t = ray_tmin;
  int iter = 0;
  float lo_bound = 0;
  float hi_bound = 4. * M_PI;

  float ld = len(ray_D);

  static bool exceeded_max = false;
  while (t < ray_tmax) {
    float3 pos = ray_P + t * ray_D;

    // If we've exceeded the maximum number of iterations,
    // print a warning
    if (iter > max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf("Warning: exceeded maximum number of Harnack iterations.\n");
      }

      return false;
    }

    float val = solid_angle(pos);

    // Calculate the radius of a ball around the current point over which
    // we know the function is harmonic.  An easy way to identify such a
    // ball is to restrict to the sphere touching the closest point on the
    // polygon boundary.
    float3 closestPoint;
    float R = distance_to_boundary(pos, &closestPoint);

    // If we're close enough to the level set, or we've exceeded the
    // maximum number of iterations, assume there's a hit.
    if (close_to_zero(val) || R < epsilon || iter > max_iterations) {
      // if (R < epsilon)   grad = pos - closestPoint; // TODO: this?
      /* std::string info = "hit info | val: " + std::to_string(val) + "\n"; */
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
    float r = get_max_step(val, R, lo_bound, hi_bound, shift);
    t += r / ld;
    iter++;
  }

  return false;
}

CCL_NAMESPACE_END

#endif /* __UTIL_MATH_INTERSECT_H__ */
