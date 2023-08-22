/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <optional>

#include "blender/session.h"
#include "blender/sync.h"
#include "blender/util.h"

#include "scene/camera.h"
#include "scene/colorspace.h"
#include "scene/nonplanar_polygon.h"
#include "scene/object.h"
#include "scene/scene.h"

#include "subd/patch.h"
#include "subd/split.h"

#include "util/algorithm.h"
#include "util/color.h"
#include "util/debug.h"
#include "util/disjoint_set.h"
#include "util/foreach.h"
#include "util/hash.h"
#include "util/log.h"
#include "util/math.h"

#include "mikktspace.hh"

#include "DNA_mesh_types.h"

CCL_NAMESPACE_BEGIN

template<typename TypeInCycles, typename GetValueAtIndex>
static void fill_generic_attribute(BL::Mesh &b_mesh,
                                   TypeInCycles *data,
                                   const BL::Attribute::domain_enum b_domain,
                                   const bool subdivision,
                                   const GetValueAtIndex &get_value_at_index)
{
  switch (b_domain) {
    case BL::Attribute::domain_CORNER: {
      if (subdivision) {
        const int polys_num = b_mesh.polygons.length();
        if (polys_num == 0) {
          return;
        }
        const int *face_offsets = static_cast<const int *>(b_mesh.polygons[0].ptr.data);
        for (int i = 0; i < polys_num; i++) {
          const int poly_start = face_offsets[i];
          const int poly_size = face_offsets[i + 1] - poly_start;
          for (int j = 0; j < poly_size; j++) {
            *data = get_value_at_index(poly_start + j);
            data++;
          }
        }
      }
      else {
        const int tris_num = b_mesh.loop_triangles.length();
        const MLoopTri *looptris = static_cast<const MLoopTri *>(
            b_mesh.loop_triangles[0].ptr.data);
        for (int i = 0; i < tris_num; i++) {
          const MLoopTri &tri = looptris[i];
          data[i * 3 + 0] = get_value_at_index(tri.tri[0]);
          data[i * 3 + 1] = get_value_at_index(tri.tri[1]);
          data[i * 3 + 2] = get_value_at_index(tri.tri[2]);
        }
      }
      break;
    }
    case BL::Attribute::domain_EDGE: {
      const size_t edges_num = b_mesh.edges.length();
      if (edges_num == 0) {
        return;
      }
      if constexpr (std::is_same_v<TypeInCycles, uchar4>) {
        /* uchar4 edge attributes do not exist, and averaging in place
         * would not work. */
        assert(0);
      }
      else {
        const int2 *edges = static_cast<const int2 *>(b_mesh.edges[0].ptr.data);
        const size_t verts_num = b_mesh.vertices.length();
        vector<int> count(verts_num, 0);

        /* Average edge attributes at vertices. */
        for (int i = 0; i < edges_num; i++) {
          TypeInCycles value = get_value_at_index(i);

          const int2 &b_edge = edges[i];
          data[b_edge[0]] += value;
          data[b_edge[1]] += value;
          count[b_edge[0]]++;
          count[b_edge[1]]++;
        }

        for (size_t i = 0; i < verts_num; i++) {
          if (count[i] > 1) {
            data[i] /= (float)count[i];
          }
        }
      }
      break;
    }
    case BL::Attribute::domain_POINT: {
      const int num_verts = b_mesh.vertices.length();
      for (int i = 0; i < num_verts; i++) {
        data[i] = get_value_at_index(i);
      }
      break;
    }
    case BL::Attribute::domain_FACE: {
      if (subdivision) {
        const int num_polygons = b_mesh.polygons.length();
        for (int i = 0; i < num_polygons; i++) {
          data[i] = get_value_at_index(i);
        }
      }
      else {
        const int tris_num = b_mesh.loop_triangles.length();
        const int *looptri_faces = static_cast<const int *>(
            b_mesh.loop_triangle_polygons[0].ptr.data);
        for (int i = 0; i < tris_num; i++) {
          data[i] = get_value_at_index(looptri_faces[i]);
        }
      }
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

static void attr_create_generic(Scene *scene,
                                NonplanarPolygon *nonplanar_polygon,
                                BL::Mesh &b_mesh)
{
  AttributeSet &attributes = nonplanar_polygon->attributes;
  const ustring default_color_name{b_mesh.attributes.default_color_name().c_str()};

  for (BL::Attribute &b_attribute : b_mesh.attributes) {
    const ustring name{b_attribute.name().c_str()};
    const bool is_render_color = name == default_color_name;

    if (!(nonplanar_polygon->need_attribute(scene, name) ||
          (is_render_color && nonplanar_polygon->need_attribute(scene, ATTR_STD_VERTEX_COLOR))))
    {
      continue;
    }
    if (attributes.find(name)) {
      continue;
    }

    const BL::Attribute::domain_enum b_domain = b_attribute.domain();
    const BL::Attribute::data_type_enum b_data_type = b_attribute.data_type();
    bool subdivision = false;

    AttributeElement element = ATTR_ELEMENT_NONE;
    switch (b_domain) {
      case BL::Attribute::domain_CORNER:
        element = ATTR_ELEMENT_CORNER;
        break;
      case BL::Attribute::domain_POINT:
        element = ATTR_ELEMENT_VERTEX;
        break;
      case BL::Attribute::domain_EDGE:
        element = ATTR_ELEMENT_VERTEX;
        break;
      case BL::Attribute::domain_FACE:
        element = ATTR_ELEMENT_FACE;
        break;
      default:
        break;
    }
    if (element == ATTR_ELEMENT_NONE) {
      /* Not supported. */
      continue;
    }
    switch (b_data_type) {
      case BL::Attribute::data_type_FLOAT: {
        BL::FloatAttribute b_float_attribute{b_attribute};
        if (b_float_attribute.data.length() == 0) {
          continue;
        }
        const float *src = static_cast<const float *>(b_float_attribute.data[0].ptr.data);
        Attribute *attr = attributes.add(name, TypeFloat, element);
        float *data = attr->data_float();
        fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) { return src[i]; });
        break;
      }
      case BL::Attribute::data_type_BOOLEAN: {
        BL::BoolAttribute b_bool_attribute{b_attribute};
        if (b_bool_attribute.data.length() == 0) {
          continue;
        }
        const bool *src = static_cast<const bool *>(b_bool_attribute.data[0].ptr.data);
        Attribute *attr = attributes.add(name, TypeFloat, element);
        float *data = attr->data_float();
        fill_generic_attribute(
            b_mesh, data, b_domain, subdivision, [&](int i) { return (float)src[i]; });
        break;
      }
      case BL::Attribute::data_type_INT: {
        BL::IntAttribute b_int_attribute{b_attribute};
        if (b_int_attribute.data.length() == 0) {
          continue;
        }
        const int *src = static_cast<const int *>(b_int_attribute.data[0].ptr.data);
        Attribute *attr = attributes.add(name, TypeFloat, element);
        float *data = attr->data_float();
        fill_generic_attribute(
            b_mesh, data, b_domain, subdivision, [&](int i) { return (float)src[i]; });
        break;
      }
      case BL::Attribute::data_type_FLOAT_VECTOR: {
        BL::FloatVectorAttribute b_vector_attribute{b_attribute};
        if (b_vector_attribute.data.length() == 0) {
          continue;
        }
        const float(*src)[3] = static_cast<const float(*)[3]>(b_vector_attribute.data[0].ptr.data);
        Attribute *attr = attributes.add(name, TypeVector, element);
        float3 *data = attr->data_float3();
        fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) {
          return make_float3(src[i][0], src[i][1], src[i][2]);
        });
        break;
      }
      case BL::Attribute::data_type_BYTE_COLOR: {
        BL::ByteColorAttribute b_color_attribute{b_attribute};
        if (b_color_attribute.data.length() == 0) {
          continue;
        }
        const uchar(*src)[4] = static_cast<const uchar(*)[4]>(b_color_attribute.data[0].ptr.data);

        if (element == ATTR_ELEMENT_CORNER) {
          element = ATTR_ELEMENT_CORNER_BYTE;
        }
        Attribute *attr = attributes.add(name, TypeRGBA, element);
        if (is_render_color) {
          attr->std = ATTR_STD_VERTEX_COLOR;
        }

        if (element == ATTR_ELEMENT_CORNER_BYTE) {
          uchar4 *data = attr->data_uchar4();
          fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) {
            /* Compress/encode vertex color using the sRGB curve. */
            return make_uchar4(src[i][0], src[i][1], src[i][2], src[i][3]);
          });
        }
        else {
          float4 *data = attr->data_float4();
          fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) {
            return make_float4(color_srgb_to_linear(byte_to_float(src[i][0])),
                               color_srgb_to_linear(byte_to_float(src[i][1])),
                               color_srgb_to_linear(byte_to_float(src[i][2])),
                               color_srgb_to_linear(byte_to_float(src[i][3])));
          });
        }
        break;
      }
      case BL::Attribute::data_type_FLOAT_COLOR: {
        BL::FloatColorAttribute b_color_attribute{b_attribute};
        if (b_color_attribute.data.length() == 0) {
          continue;
        }
        const float(*src)[4] = static_cast<const float(*)[4]>(b_color_attribute.data[0].ptr.data);

        Attribute *attr = attributes.add(name, TypeRGBA, element);
        if (is_render_color) {
          attr->std = ATTR_STD_VERTEX_COLOR;
        }

        float4 *data = attr->data_float4();
        fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) {
          return make_float4(src[i][0], src[i][1], src[i][2], src[i][3]);
        });
        break;
      }
      case BL::Attribute::data_type_FLOAT2: {
        BL::Float2Attribute b_float2_attribute{b_attribute};
        if (b_float2_attribute.data.length() == 0) {
          continue;
        }
        const float(*src)[2] = static_cast<const float(*)[2]>(b_float2_attribute.data[0].ptr.data);
        Attribute *attr = attributes.add(name, TypeFloat2, element);
        float2 *data = attr->data_float2();
        fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) {
          return make_float2(src[i][0], src[i][1]);
        });
        break;
      }
      case BL::Attribute::data_type_INT32_2D: {
        BL::Int2Attribute b_int2_attribute{b_attribute};
        if (b_int2_attribute.data.length() == 0) {
          continue;
        }
        const int2 *src = static_cast<const int2 *>(b_int2_attribute.data[0].ptr.data);
        Attribute *attr = attributes.add(name, TypeFloat2, element);
        float2 *data = attr->data_float2();
        fill_generic_attribute(b_mesh, data, b_domain, subdivision, [&](int i) {
          return make_float2(float(src[i][0]), float(src[i][1]));
        });
        break;
      }
      default:
        /* Not supported. */
        break;
    }
  }
}

/* Create nonplanar polygon */

static const int *find_corner_vert_attribute(BL::Mesh b_mesh)
{
  for (BL::Attribute &b_attribute : b_mesh.attributes) {
    if (b_attribute.domain() != BL::Attribute::domain_CORNER) {
      continue;
    }
    if (b_attribute.data_type() != BL::Attribute::data_type_INT) {
      continue;
    }
    if (b_attribute.name() != ".corner_vert") {
      continue;
    }
    BL::IntAttribute b_int_attribute{b_attribute};
    if (b_int_attribute.data.length() == 0) {
      return nullptr;
    }
    return static_cast<const int *>(b_int_attribute.data[0].ptr.data);
  }
  return nullptr;
}

static const int *find_material_index_attribute(BL::Mesh b_mesh)
{
  for (BL::Attribute &b_attribute : b_mesh.attributes) {
    if (b_attribute.domain() != BL::Attribute::domain_FACE) {
      continue;
    }
    if (b_attribute.data_type() != BL::Attribute::data_type_INT) {
      continue;
    }
    if (b_attribute.name() != "material_index") {
      continue;
    }
    BL::IntAttribute b_int_attribute{b_attribute};
    if (b_int_attribute.data.length() == 0) {
      return nullptr;
    }
    return static_cast<const int *>(b_int_attribute.data[0].ptr.data);
  }
  return nullptr;
}

static void create_nonplanar_polygon(Scene *scene,
                                     NonplanarPolygon *nonplanar_polygon,
                                     BL::Mesh &b_mesh,
                                     const array<Node *> &used_shaders,
                                     const bool need_motion,
                                     const float motion_scale,
                                     const bool subdivision = false,
                                     const bool subdivide_uvs = true)
{
  int numfaces = b_mesh.polygons.length();

  /* If no faces, create empty nonplanar polygon. */
  if (numfaces == 0) {
    return;
  }

  numfaces = 1;  // only keep one face

  const float(*positions)[3] = static_cast<const float(*)[3]>(b_mesh.vertices[0].ptr.data);
  const int *face_offsets = static_cast<const int *>(b_mesh.polygons[0].ptr.data);
  const int *corner_verts = find_corner_vert_attribute(b_mesh);

  const int numverts = face_offsets[1] - face_offsets[0];

  /* allocate memory */
  nonplanar_polygon->resize_nonplanar_polygon(numverts);

  float3 *verts = nonplanar_polygon->get_verts().data();
  for (int iV = 0; iV < numverts; iV++) {
    int v = corner_verts[face_offsets[0] + iV];
    verts[iV] = make_float3(positions[v][0], positions[v][1], positions[v][2]);
  }

  AttributeSet &attributes = nonplanar_polygon->attributes;

  auto clamp_material_index = [&](const int material_index) -> int {
    return clamp(material_index, 0, used_shaders.size() - 1);
  };

  int *shader = nonplanar_polygon->get_shader().data();

  const int *material_indices = find_material_index_attribute(b_mesh);
  if (material_indices) {
    const int *looptri_faces = static_cast<const int *>(b_mesh.loop_triangle_polygons[0].ptr.data);
    shader[0] = clamp_material_index(material_indices[looptri_faces[0]]);
  }
  else {
    std::fill(shader, shader + 1, 0);
  }

  nonplanar_polygon->tag_shader_modified();

  /* Create all needed attributes.
   * The calculate functions will check whether they're needed or not.
   */
  attr_create_generic(scene, nonplanar_polygon, b_mesh);
}

/* Sync */

void BlenderSync::sync_nonplanar_polygon(BL::Depsgraph b_depsgraph,
                                         BObjectInfo &b_ob_info,
                                         NonplanarPolygon *nonplanar_polygon)
{
  /* make a copy of the shaders as the caller in the main thread still need them for syncing the
   * attributes */
  array<Node *> used_shaders = nonplanar_polygon->get_used_shaders();

  NonplanarPolygon new_nonplanar_polygon;
  new_nonplanar_polygon.set_used_shaders(used_shaders);

  if (view_layer.use_surfaces) {

    /* For some reason, meshes do not need this... */
    bool need_undeformed = new_nonplanar_polygon.need_attribute(scene, ATTR_STD_GENERATED);
    BL::Mesh b_mesh = object_to_mesh(
        b_data, b_ob_info, b_depsgraph, need_undeformed, Mesh::SUBDIVISION_NONE);

    if (b_mesh) {
      const bool need_motion = false;
      const float motion_scale = 0.f;

      /* Sync nonplanar_polygon itself. */
      create_nonplanar_polygon(scene,
                               &new_nonplanar_polygon,
                               b_mesh,
                               new_nonplanar_polygon.get_used_shaders(),
                               need_motion,
                               motion_scale,
                               false);

      free_object_to_mesh(b_data, b_ob_info, b_mesh);
    }
  }

  /* update original sockets */

  nonplanar_polygon->clear_non_sockets();

  for (const SocketType &socket : new_nonplanar_polygon.type->inputs) {
    /* Those sockets are updated in sync_object, so do not modify them. */
    if (socket.name == "use_motion_blur" || socket.name == "motion_steps" ||
        socket.name == "used_shaders")
    {
      continue;
    }
    nonplanar_polygon->set_value(socket, new_nonplanar_polygon, socket);
  }

  nonplanar_polygon->attributes.update(std::move(new_nonplanar_polygon.attributes));

  /* tag update */
  bool rebuild = false;  // TODO: need to check something

  nonplanar_polygon->tag_update(scene, rebuild);
}

CCL_NAMESPACE_END
