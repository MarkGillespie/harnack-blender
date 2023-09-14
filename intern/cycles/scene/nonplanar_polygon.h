/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef __NONPLANAR_POLYGON_H__
#define __NONPLANAR_POLYGON_H__

#include "graph/node.h"

#include "bvh/params.h"
#include "scene/attribute.h"
#include "scene/geometry.h"
#include "scene/shader.h"

#include "util/array.h"
#include "util/boundbox.h"
#include "util/list.h"
#include "util/map.h"
#include "util/param.h"
#include "util/set.h"
#include "util/types.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Attribute;
class BVH;
class Device;
class DeviceScene;
class Progress;
class RenderStats;
class Scene;
class SceneParams;
class AttributeRequest;
struct SubdParams;
class DiagSplit;
struct PackedPatchTable;

/* Nonplanar polygon mesh */

class NonplanarPolygonMesh : public Geometry {
 protected:
  NonplanarPolygonMesh(const NodeType *node_type_, Type geom_type_);

 public:
  NODE_DECLARE

  /* NonplanarPolygon Data */
  NODE_SOCKET_API_ARRAY(array<float3>, verts)
  NODE_SOCKET_API_ARRAY(array<int>, face_starts)
  NODE_SOCKET_API_ARRAY(array<int>, face_sizes)
  NODE_SOCKET_API_ARRAY(array<int>, shader)
  NODE_SOCKET_API(int, scenario);
  NODE_SOCKET_API(float, epsilon);
  NODE_SOCKET_API(float, levelset);
  NODE_SOCKET_API(float, frequency);
  NODE_SOCKET_API(float, boundingbox_expansion);
  NODE_SOCKET_API(int, max_iterations);
  NODE_SOCKET_API(int, precision);
  NODE_SOCKET_API(bool, use_grad_termination);
  NODE_SOCKET_API(bool, polygon_with_holes);
  NODE_SOCKET_API(bool, clip_y);

  // from DNA_modifier_types.h
  // MOD_HARNACK_TRIANGULATE = 0,
  // MOD_HARNACK_PREQUANTUM = 1,
  // MOD_HARNACK_GAUSS_BONNET = 2,
  NODE_SOCKET_API(int, solid_angle_formula);

  size_t prim_space() const
  {
    return face_starts.size();
  }

  size_t vert_space() const
  {
    // store a center point per face, and a set of harnack parameters per face
    return verts.size() + 3 * face_starts.size();
  }

  size_t num_faces() const
  {
    return face_starts.size();
  }

  size_t num_corners() const
  {
    return verts.size();
  }

 private:
  /* BVH */
  size_t vert_offset;

  unordered_map<int, int> vert_to_stitching_key_map; /* real vert index -> stitching index */
  unordered_multimap<int, int>
      vert_stitching_map; /* stitching index -> multiple real vert indices */

  friend class BVH2;
  friend class BVHBuild;
  friend class BVHSpatialSplit;
  friend class DiagSplit;
  friend class EdgeDice;
  friend class GeometryManager;
  friend class ObjectManager;

 public:
  /* Functions */
  NonplanarPolygonMesh();
  ~NonplanarPolygonMesh();

  void resize_nonplanar_polygon(int numverts, int numfaces);
  void reserve_nonplanar_polygon(int numverts, int numfaces);
  void clear_non_sockets();
  void clear(bool preserve_shaders = false) override;
  void add_vertex(float3 P);
  void add_vertex_slow(float3 P);

  void copy_center_to_motion_step(const int motion_step);

  void compute_bounds() override;
  BoundBox compute_face_bounds(size_t iF) const;
  void apply_transform(const Transform &tfm, const bool apply_to_motion) override;
  void add_undisplaced();

  void pack_shaders(Scene *scene, uint *shader);
  void pack_verts(packed_float3 *tri_verts, packed_uint3 *tri_vindex);

  void get_uv_tiles(ustring map, unordered_set<int> &tiles) override;

  PrimitiveType primitive_type() const override;

 protected:
  void clear(bool preserve_shaders, bool preserve_voxel_data);
};

CCL_NAMESPACE_END

#endif /* __NONPLANAR_POLYGON_H__ */
