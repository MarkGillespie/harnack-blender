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

#include "harnack.ipp"

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
  uint mode = polygon_data[0].z;
  uint scenario = mode >> 5;
  uint intersection_mode = (mode >> 2) & 7;
  uint gradient_mode = mode & 3;

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
      sa_params.clip_y = properties & 1;
      params = sa_params.pts[N + 2];
      sa_params.frequency = params.x;
      uint acc_cap = static_cast<uint>(params.y);
      sa_params.use_quick_triangulation = (acc_cap >> 4) & 1;
      sa_params.use_grad_termination = (acc_cap >> 3) & 1;
      sa_params.use_overstepping = (acc_cap >> 2) & 1;
      sa_params.use_newton = (acc_cap >> 1) & 1;
      sa_params.use_extrapolation = false;
      sa_params.fixed_step_count = false;
      sa_params.capture_misses = acc_cap & 1;
      sa_params.n_loops = static_cast<uint>(params.z);
      switch (intersection_mode) {
        case MOD_HARNACK_HARNACK: {
          if (precision == 0) {
            found_intersection = ray_nonplanar_polygon_intersect_T<float>(sa_params, &u, &v, &t);
          }
          else if (precision == 1) {
            found_intersection = ray_nonplanar_polygon_intersect_T<double>(sa_params, &u, &v, &t);
          }
          break;
        }
        case MOD_HARNACK_NEWTON: {
          if (precision == 0) {
            found_intersection = newton_intersect_T<float>(sa_params, &u, &v, &t);
          }
          else if (precision == 1) {
            found_intersection = newton_intersect_T<double>(sa_params, &u, &v, &t);
          }
          break;
        }
        case MOD_HARNACK_BISECTION: {
          if (precision == 0) {
            found_intersection = bisection_intersect_T<float>(sa_params, &u, &v, &t);
          }
          else if (precision == 1) {
            found_intersection = bisection_intersect_T<double>(sa_params, &u, &v, &t);
          }
          break;
        }
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
    case MOD_HARNACK_GYROID: {  // Riemann surface
      const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

      gyroid_intersection_params g_params;
      g_params.ray_P = P;
      g_params.ray_D = dir;
      g_params.ray_tmin = tmin;
      g_params.ray_tmax = tmax;

      g_params.epsilon = pts[0].x;
      g_params.levelset = pts[0].y;
      uint properties = static_cast<uint>(pts[0].z);
      g_params.max_iterations = properties >> 6;
      g_params.use_grad_termination = properties & 1;
      uint precision = (properties >> 1) & 1;

      g_params.R = pts[1].x;
      g_params.frequency = pts[1].y;

      uint acc_cap = static_cast<uint>(pts[1].z);
      g_params.use_grad_termination = (acc_cap >> 3) & 1;
      g_params.use_overstepping = (acc_cap >> 2) & 1;

      switch (intersection_mode) {
        case MOD_HARNACK_HARNACK: {
          if (precision == 0) {
            found_intersection = ray_gyroid_intersect_T<float>(g_params, &u, &v, &t);
          }
          else if (precision == 1) {
            found_intersection = ray_gyroid_intersect_T<double>(g_params, &u, &v, &t);
          }
          break;
        }
        case MOD_HARNACK_NEWTON: {
          if (precision == 0) {
            found_intersection = newton_intersect_gyroid_T<float>(g_params, &u, &v, &t);
          }
          else if (precision == 1) {
            found_intersection = newton_intersect_gyroid_T<double>(g_params, &u, &v, &t);
          }
          break;
        }
        case MOD_HARNACK_BISECTION: {
          if (precision == 0) {
            found_intersection = bisection_intersect_gyroid_T<float>(g_params, &u, &v, &t);
          }
          else if (precision == 1) {
            found_intersection = bisection_intersect_gyroid_T<double>(g_params, &u, &v, &t);
          }
          break;
        }
      }
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

  uint mode = polygon_data[0].z;
  uint scenario = mode >> 5;
  uint intersection_mode = (mode >> 2) & 7;
  uint gradient_mode = mode & 3;
  const packed_float3 *pts = &kernel_data_fetch(tri_verts, polygon_start);

  float3 params = pts[N + 2];
  uint acc_cap = static_cast<uint>(params.y);
  bool capture_misses = acc_cap & (1 << 0);

  float3 P = ray->P + isect->t * ray->D;
  float3 normal;

  if (capture_misses) {
    normal = normalize(-ray->D);
  }
  else {
    switch (scenario) {
      case MOD_HARNACK_NONPLANAR_POLYGON: {
        uint n_loops = static_cast<uint>(pts[N + 2].z);
        normal = ray_nonplanar_polygon_normal_T<double>(
            P, polygon_data, pts, n_loops, gradient_mode);
        break;
      }
      case MOD_HARNACK_DISK_SHELL: {  // disk shell
        // TODO
        break;
      }
      case MOD_HARNACK_SPHERICAL_HARMONIC: {  // spherical harmonic
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
      case MOD_HARNACK_GYROID: {  // Gyroid
        float R = pts[1].x;
        float frequency = pts[1].y;
        normal = ray_gyroid_normal_T<double>(P, R, frequency);
        break;
      }
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
