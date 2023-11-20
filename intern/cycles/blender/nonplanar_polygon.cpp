/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <optional>

#include "blender/session.h"
#include "blender/sync.h"
#include "blender/util.h"

#include "BKE_attribute.hh"

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

static void attr_create_generic(Scene *scene, NonplanarPolygonMesh *mesh, BL::Mesh &b_mesh)
{
  AttributeSet &attributes = mesh->attributes;
  const ustring default_color_name{b_mesh.attributes.default_color_name().c_str()};

  for (BL::Attribute &b_attribute : b_mesh.attributes) {
    const ustring name{b_attribute.name().c_str()};
    const bool is_render_color = name == default_color_name;

    if (!(mesh->need_attribute(scene, name) ||
          (is_render_color && mesh->need_attribute(scene, ATTR_STD_VERTEX_COLOR))))
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

static void create_nonplanar_polygon_mesh(Scene *scene,
                                          NonplanarPolygonMesh *mesh,
                                          BL::Mesh &b_mesh,
                                          const array<Node *> &used_shaders)
{
  int numfaces = b_mesh.polygons.length();

  // If no faces, create empty nonplanar polygon.
  if (numfaces == 0) {
    return;
  }

  //== extract mesh geometry
  const float(*positions)[3] = static_cast<const float(*)[3]>(b_mesh.vertices[0].ptr.data);
  const int *face_offsets = static_cast<const int *>(b_mesh.polygons[0].ptr.data);
  const int *corner_verts = find_corner_vert_attribute(b_mesh);

  // store each corner separately
  int numverts = 0;
  for (uint j = 0; j < numfaces; j++) {
    int face_size = face_offsets[j + 1] - face_offsets[j];
    numverts += face_size;
  }

  // allocate memory
  mesh->resize_nonplanar_polygon(numverts, numfaces);

  float3 *verts = mesh->get_verts().data();
  int *face_starts = mesh->get_face_starts().data();
  int *face_sizes = mesh->get_face_sizes().data();
  uint iV = 0;
  for (uint j = 0; j < numfaces; j++) {
    int face_size = face_offsets[j + 1] - face_offsets[j];
    for (int i = 0; i < face_size; i++) {
      int v = corner_verts[face_offsets[j] + i];
      verts[iV] = make_float3(positions[v][0], positions[v][1], positions[v][2]);
      iV++;
    }
    face_starts[j] = face_offsets[j];
    face_sizes[j] = face_size;
  }

  AttributeSet &attributes = mesh->attributes;

  auto clamp_material_index = [&](const int material_index) -> int {
    return clamp(material_index, 0, used_shaders.size() - 1);
  };

  // assign shaders
  int *shader = mesh->get_shader().data();

  const int *material_indices = find_material_index_attribute(b_mesh);
  if (material_indices) {
    for (int i = 0; i < numfaces; i++) {
      shader[i] = clamp_material_index(material_indices[i]);
    }
  }
  else {
    std::fill(shader, shader + numfaces, 0);
  }

  mesh->tag_shader_modified();

  /* Create all needed attributes.
   * The calculate functions will check whether they're needed or not.
   */
  attr_create_generic(scene, mesh, b_mesh);

  //== read off parameter values from attributes if available
  mesh->set_epsilon(0.0001);
  mesh->set_levelset(0.5);
  mesh->set_boundingbox_expansion(0);

  // I don't know why I can't find these attributes using b_mesh.attributes["EPSILON"], but looping
  // through all the attributes seems to work fine
  std::string epsilon_tag = "EPSILON", levelset_tag = "LEVELSET", bounds_tag = "BOUNDS",
              grad_termination_tag = "GRAD_TERMINATION", formula_tag = "SAF", holes_tag = "HOLES",
              iteration_tag = "MAX_ITERATIONS", precision_tag = "PRECISION", clip_tag = "CLIP",
              frequency_tag = "FREQUENCY", r_tag = "R", l_tag = "L", m_tag = "M",
              capture_misses_tag = "CAPTURE_MISSES", harnack_tag = "HARNACK";
  for (BL::Attribute &b_attribute : b_mesh.attributes) {
    const ustring name{b_attribute.name().c_str()};

    if (name == harnack_tag) {
      BL::FloatAttribute harnack_attribute{b_attribute};
      const float *harnack_data = static_cast<const float *>(harnack_attribute.data[0].ptr.data);
      mesh->set_scenario(static_cast<int>(harnack_data[0]));
    }
    if (name == epsilon_tag) {
      BL::FloatAttribute epsilon_attribute{b_attribute};
      const float *epsilon_data = static_cast<const float *>(epsilon_attribute.data[0].ptr.data);
      mesh->set_epsilon(epsilon_data[0]);
    }
    else if (name == levelset_tag) {
      BL::FloatAttribute levelset_attribute{b_attribute};
      const float *levelset_data = static_cast<const float *>(levelset_attribute.data[0].ptr.data);
      mesh->set_levelset(levelset_data[0]);
    }
    else if (name == frequency_tag) {
      BL::FloatAttribute frequency_attribute{b_attribute};
      const float *frequency_data = static_cast<const float *>(
          frequency_attribute.data[0].ptr.data);
      mesh->set_frequency(frequency_data[0]);
    }
    else if (name == bounds_tag) {
      BL::FloatAttribute bounds_attribute{b_attribute};
      const float *bounds_data = static_cast<const float *>(bounds_attribute.data[0].ptr.data);
      mesh->set_boundingbox_expansion(bounds_data[0]);
    }
    else if (name == grad_termination_tag) {
      mesh->set_use_grad_termination(true);
    }
    else if (name == formula_tag) {
      BL::FloatAttribute formula_attribute{b_attribute};
      const float *formula_data = static_cast<const float *>(formula_attribute.data[0].ptr.data);
      mesh->set_solid_angle_formula(static_cast<int>(formula_data[0]));
    }
    else if (name == holes_tag) {
      mesh->set_polygon_with_holes(true);
    }
    else if (name == capture_misses_tag) {
      mesh->set_capture_misses(true);
    }
    else if (name == iteration_tag) {
      BL::FloatAttribute iteration_attribute{b_attribute};
      const float *iteration_data = static_cast<const float *>(
          iteration_attribute.data[0].ptr.data);
      mesh->set_max_iterations(static_cast<int>(iteration_data[0]));
    }
    else if (name == precision_tag) {
      BL::FloatAttribute precision_attribute{b_attribute};
      const float *precision_data = static_cast<const float *>(
          precision_attribute.data[0].ptr.data);
      mesh->set_precision(static_cast<int>(precision_data[0]));
    }
    else if (name == clip_tag) {
      mesh->set_clip_y(true);
    }
    else if (name == r_tag) {
      BL::FloatAttribute R_attribute{b_attribute};
      const float *R_data = static_cast<const float *>(R_attribute.data[0].ptr.data);
      mesh->set_R(R_data[0]);
    }
    else if (name == l_tag) {
      BL::FloatAttribute l_attribute{b_attribute};
      const float *l_data = static_cast<const float *>(l_attribute.data[0].ptr.data);
      mesh->set_l(static_cast<int>(l_data[0]));
    }
    else if (name == m_tag) {
      BL::FloatAttribute m_attribute{b_attribute};
      const float *m_data = static_cast<const float *>(m_attribute.data[0].ptr.data);
      mesh->set_m(static_cast<int>(m_data[0]));
    }
  }
}

/* Sync */

void BlenderSync::sync_nonplanar_polygon_mesh(BL::Depsgraph b_depsgraph,
                                              BObjectInfo &b_ob_info,
                                              NonplanarPolygonMesh *mesh)
{
  /* make a copy of the shaders as the caller in the main thread still need them for syncing the
   * attributes */
  array<Node *> used_shaders = mesh->get_used_shaders();

  NonplanarPolygonMesh new_mesh;
  new_mesh.set_used_shaders(used_shaders);

  if (view_layer.use_surfaces) {

    // For some reason, meshes do not need this...
    bool need_undeformed = new_mesh.need_attribute(scene, ATTR_STD_GENERATED);
    BL::Mesh b_mesh = object_to_mesh(
        b_data, b_ob_info, b_depsgraph, need_undeformed, Mesh::SUBDIVISION_NONE);

    if (b_mesh) {
      // Sync mesh itself.
      create_nonplanar_polygon_mesh(scene, &new_mesh, b_mesh, new_mesh.get_used_shaders());

      free_object_to_mesh(b_data, b_ob_info, b_mesh);
    }
  }

  /* update original sockets */

  mesh->clear_non_sockets();

  for (const SocketType &socket : new_mesh.type->inputs) {
    /* Those sockets are updated in sync_object, so do not modify them. */
    if (socket.name == "use_motion_blur" || socket.name == "motion_steps" ||
        socket.name == "used_shaders")
    {
      continue;
    }
    mesh->set_value(socket, new_mesh, socket);
  }

  mesh->attributes.update(std::move(new_mesh.attributes));

  /* tag update */
  bool rebuild = (mesh->face_starts_is_modified());

  mesh->tag_update(scene, rebuild);
}

CCL_NAMESPACE_END
