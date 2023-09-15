/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Nonplanar Polygon Primitive
 *
 * Basic nonplanar_polygon with 3 vertices is used to represent mesh surfaces. For BVH
 * ray intersection we use a precomputed nonplanar_polygon storage to accelerate
 * intersection at the cost of more memory usage */

#pragma once

CCL_NAMESPACE_BEGIN

// Normal on nonplanar_polygon.
// TODO: move from nonplanar_polygon_intersect.h?

// ccl_device_inline float3 nonplanar_polygon_normal(KernelGlobals kg, ccl_private ShaderData *sd)
//
// TODO: implement other normal utils in triangle.h?

/* Reading attributes on various nonplanar_polygon elements */

// TODO: figure out interpolation
// ccl_device float nonplanar_polygon_attribute_float(KernelGlobals kg,
//                                                    ccl_private const ShaderData *sd,
//                                                    const AttributeDescriptor desc,
//                                                    ccl_private float *dx,
//                                                    ccl_private float *dy)

// ccl_device float3 nonplanar_polygon_attribute_float3(KernelGlobals kg,
//                                                      ccl_private const ShaderData *sd,
//                                                      const AttributeDescriptor desc,
//                                                      ccl_private float3 *dx,
//                                                      ccl_private float3 *dy)

ccl_device float2 nonplanar_polygon_attribute_float2(KernelGlobals kg,
                                                     ccl_private const ShaderData *sd,
                                                     const AttributeDescriptor desc,
                                                     ccl_private float2 *dx,
                                                     ccl_private float2 *dy)
{
  // HACK to access UVs from shaders
  return make_float2(sd->u, sd->v);
}

CCL_NAMESPACE_END
