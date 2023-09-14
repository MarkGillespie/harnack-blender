/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "bvh/build.h"
#include "bvh/bvh.h"

#include "device/device.h"

#include "scene/hair.h"
#include "scene/nonplanar_polygon.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader_graph.h"

#include "util/foreach.h"
#include "util/log.h"
#include "util/progress.h"
#include "util/set.h"

CCL_NAMESPACE_BEGIN

/* NonplanarPolygonMesh */

NODE_DEFINE(NonplanarPolygonMesh)
{
  NodeType *type = NodeType::add(
      "Nonplanar Polygon", create, NodeType::NONE, Geometry::get_node_base_type());

  SOCKET_INT(scenario, "Scenario", 0);
  SOCKET_POINT_ARRAY(verts, "Vertices", array<float3>());
  SOCKET_INT_ARRAY(face_starts, "Face Starts", array<int>());
  SOCKET_INT_ARRAY(face_sizes, "Face Sizes", array<int>());
  SOCKET_INT_ARRAY(shader, "Shader", array<int>());
  SOCKET_FLOAT(epsilon, "Epsilon", 0);
  SOCKET_FLOAT(levelset, "Levelset", 0);
  SOCKET_FLOAT(frequency, "Frequency", 0);
  SOCKET_FLOAT(boundingbox_expansion, "Bounding Box Expansion", 0);
  SOCKET_INT(solid_angle_formula, "Solid Angle Formula", 0);
  SOCKET_INT(max_iterations, "Max Iterations", 1500);
  SOCKET_INT(precision, "Precision", 1);
  SOCKET_BOOLEAN(use_grad_termination, "Use Gradient Termination Condition", false);
  SOCKET_BOOLEAN(polygon_with_holes, "Polygon with Holes", false);
  SOCKET_BOOLEAN(clip_y, "Clip Y", false);

  return type;
}

NonplanarPolygonMesh::NonplanarPolygonMesh(const NodeType *node_type, Type geom_type_)
    : Geometry(node_type, geom_type_)
{
  vert_offset = 0;
}

NonplanarPolygonMesh::NonplanarPolygonMesh()
    : NonplanarPolygonMesh(get_node_type(), Geometry::NONPLANAR_POLYGON_MESH)
{
}

NonplanarPolygonMesh::~NonplanarPolygonMesh() {}

void NonplanarPolygonMesh::resize_nonplanar_polygon(int numverts, int numfaces)
{
  verts.resize(numverts);
  face_starts.resize(numfaces);
  face_sizes.resize(numfaces);
  shader.resize(numfaces);
  attributes.resize();
}

void NonplanarPolygonMesh::reserve_nonplanar_polygon(int numverts, int numfaces)
{
  /* reserve space to add verts later */
  verts.reserve(numverts);
  face_starts.reserve(numfaces);
  face_sizes.reserve(numfaces);
  shader.reserve(numfaces);
  attributes.resize(true);
}

void NonplanarPolygonMesh::clear_non_sockets()
{
  Geometry::clear(true);
}

void NonplanarPolygonMesh::clear(bool preserve_shaders, bool preserve_voxel_data)
{
  Geometry::clear(preserve_shaders);

  /* clear all verts */
  verts.clear();

  attributes.clear(preserve_voxel_data);

  clear_non_sockets();
}

void NonplanarPolygonMesh::clear(bool preserve_shaders)
{
  clear(preserve_shaders, false);
}

void NonplanarPolygonMesh::add_vertex(float3 P)
{
  verts.push_back_reserved(P);
  tag_verts_modified();
}

void NonplanarPolygonMesh::add_vertex_slow(float3 P)
{
  verts.push_back_slow(P);
  tag_verts_modified();
}

void NonplanarPolygonMesh::copy_center_to_motion_step(const int motion_step)
{
  Attribute *attr_mP = attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);

  if (attr_mP) {
    Attribute *attr_mN = attributes.find(ATTR_STD_MOTION_VERTEX_NORMAL);
    Attribute *attr_N = attributes.find(ATTR_STD_VERTEX_NORMAL);
    float3 *P = &verts[0];
    float3 *N = (attr_N) ? attr_N->data_float3() : NULL;
    size_t numverts = verts.size();

    memcpy(attr_mP->data_float3() + motion_step * numverts, P, sizeof(float3) * numverts);
    if (attr_mN)
      memcpy(attr_mN->data_float3() + motion_step * numverts, N, sizeof(float3) * numverts);
  }
}

void NonplanarPolygonMesh::get_uv_tiles(ustring map, unordered_set<int> &tiles)
{
  Attribute *attr;

  if (map.empty()) {
    attr = attributes.find(ATTR_STD_UV);
  }
  else {
    attr = attributes.find(map);
  }

  if (attr) {
    attr->get_uv_tiles(this, ATTR_PRIM_GEOMETRY, tiles);
  }
}

void NonplanarPolygonMesh::compute_bounds()
{
  BoundBox bnds = BoundBox::empty;
  size_t verts_size = verts.size();

  if (verts_size > 0) {
    for (size_t iF = 0; iF < num_faces(); iF++)
      bnds.grow(compute_face_bounds(iF));
  }

  if (!bnds.valid()) {
    /* empty nonplanarPolygon */
    bnds.grow(zero_float3());
  }

  bounds = bnds;
}

BoundBox NonplanarPolygonMesh::compute_face_bounds(size_t iF) const
{
  float3 center = make_float3(0, 0, 0);
  BoundBox face_bounds;
  for (size_t i = 0; i < face_sizes[iF]; i++) {
    float3 pt = verts[face_starts[iF] + i];
    face_bounds.grow(pt);
    center += pt;
  }

  if (boundingbox_expansion > 0) {
    center /= static_cast<float>(face_sizes[iF]);
    float scale = 1 + boundingbox_expansion;
    face_bounds.min = center + scale * (face_bounds.min - center);
    face_bounds.max = center + scale * (face_bounds.max - center);
  }

  return face_bounds;
}

void NonplanarPolygonMesh::apply_transform(const Transform &tfm, const bool apply_to_motion)
{
  transform_normal = transform_transposed_inverse(tfm);

  /* apply to nonplanarPolygon vertices */
  for (size_t i = 0; i < verts.size(); i++)
    verts[i] = transform_point(&tfm, verts[i]);

  tag_verts_modified();

  if (apply_to_motion) {
    Attribute *attr = attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);

    if (attr) {
      size_t steps_size = verts.size() * (motion_steps - 1);
      float3 *vert_steps = attr->data_float3();

      for (size_t i = 0; i < steps_size; i++)
        vert_steps[i] = transform_point(&tfm, vert_steps[i]);
    }

    Attribute *attr_N = attributes.find(ATTR_STD_MOTION_VERTEX_NORMAL);

    if (attr_N) {
      Transform ntfm = transform_normal;
      size_t steps_size = verts.size() * (motion_steps - 1);
      float3 *normal_steps = attr_N->data_float3();

      for (size_t i = 0; i < steps_size; i++)
        normal_steps[i] = normalize(transform_direction(&ntfm, normal_steps[i]));
    }
  }
}

void NonplanarPolygonMesh::add_undisplaced()
{
  AttributeSet &attrs = attributes;

  /* don't compute if already there */
  if (attrs.find(ATTR_STD_POSITION_UNDISPLACED)) {
    return;
  }

  /* get attribute */
  Attribute *attr = attrs.add(ATTR_STD_POSITION_UNDISPLACED);
  attr->flags |= ATTR_SUBDIVIDED;

  float3 *data = attr->data_float3();

  /* copy verts */
  size_t size = attr->buffer_size(this, ATTR_PRIM_GEOMETRY);

  if (size) {
    memcpy(data, verts.data(), size);
  }
}

void NonplanarPolygonMesh::pack_shaders(Scene *scene, uint *tri_shader)
{
  uint shader_id = 0;
  uint last_shader = -1;

  size_t triangles_size = face_starts.size();
  const int *shader_ptr = shader.data();

  for (size_t i = 0; i < triangles_size; i++) {
    const int new_shader = shader_ptr ? shader_ptr[i] : INT_MAX;

    if (new_shader != last_shader) {
      last_shader = new_shader;
      Shader *shader = (last_shader < used_shaders.size()) ?
                           static_cast<Shader *>(used_shaders[last_shader]) :
                           scene->default_surface;
      shader_id = scene->shader_manager->get_shader_id(shader, false);
    }

    tri_shader[i] = shader_id;
  }
}

void NonplanarPolygonMesh::pack_verts(packed_float3 *tri_verts, packed_uint3 *tri_vindex)
{
  size_t triangles_size = face_starts.size();
  uint properties = (max_iterations << 6) | (static_cast<uint>(solid_angle_formula) << 2) |
                    (static_cast<uint>(precision) << 1) | use_grad_termination;
  uint n_loops = polygon_with_holes ? triangles_size : 1;
  uint v_id = 0;

  for (uint j = 0; j < triangles_size; j++) {
    tri_vindex[j] = make_packed_uint3(vert_offset + v_id, face_sizes[j], n_loops);
    float3 pt_sum = make_float3(0, 0, 0);
    for (size_t i = 0; i < face_sizes[j]; i++) {
      tri_verts[v_id] = verts[face_starts[j] + i];
      pt_sum += verts[face_starts[j] + i];
      v_id++;
    }
    tri_verts[v_id] = pt_sum / face_sizes[j];
    tri_verts[v_id + 1] = make_float3(epsilon, levelset, static_cast<float>(properties));
    tri_verts[v_id + 2] = make_float3(frequency, clip_y, 0);
    v_id += 3;
  }
}

PrimitiveType NonplanarPolygonMesh::primitive_type() const
{
  return has_motion_blur() ? PRIMITIVE_MOTION_TRIANGLE : PRIMITIVE_TRIANGLE;
}

CCL_NAMESPACE_END
