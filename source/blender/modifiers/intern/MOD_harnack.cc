#include "BLT_translation.h"  // IFACE_, TIP_

#include "UI_interface.hh"
#include "UI_resources.hh"

#include "RE_engine.h"  // RenderEngineType

#include "BKE_attribute.hh"  // ATTR_DOMAIN_POINT?
#include "BKE_context.h"     // CTX_data_engine_type, CTX_data_scene
#include "BKE_mesh.hh"       // vert_positions_for_write
#include "BKE_modifier.h"
#include "BKE_scene.h"

#include "DNA_customdata_types.h"  // CD_PROP_FLOAT
#include "DNA_defaults.h"          // DNA_struct_default_get
#include "DNA_mesh_types.h"
#include "DNA_modifier_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"  // Panel
#include "RNA_access.hh"       // RNA_float_get
#include "RNA_prototypes.h"

#include "MOD_modifiertypes.hh"
#include "MOD_ui_common.hh"

static void init_data(ModifierData *md)
{
  HarnackModifierData *smd = (HarnackModifierData *)md;

  BLI_assert(MEMCMP_STRUCT_AFTER_IS_ZERO(smd, modifier));

  MEMCPY_STRUCT_AFTER(smd, DNA_struct_default_get(HarnackModifierData), modifier);
}

static Mesh *harnack_applyModifier(struct ModifierData *md,
                                   const struct ModifierEvalContext *ctx,
                                   struct Mesh *mesh)
{
  HarnackModifierData *smd = (HarnackModifierData *)md;

  std::string epsilon_tag = "EPSILON", levelset_tag = "LEVELSET", bounds_tag = "BOUNDS",
              grad_termination_tag = "GRAD_TERMINATION", formula_tag = "SAF",
              harnack_tag = "HARNACK";

  blender::bke::AttributeWriter<float> faw =
      mesh->attributes_for_write().lookup_or_add_for_write<float>(harnack_tag, ATTR_DOMAIN_POINT);
  faw.varray.set(0, 1);

  if (smd->use_grad_termination) {
    faw = mesh->attributes_for_write().lookup_or_add_for_write<float>(grad_termination_tag,
                                                                      ATTR_DOMAIN_POINT);
    faw.varray.set(0, 1);
  }

  faw = mesh->attributes_for_write().lookup_or_add_for_write<float>(epsilon_tag,
                                                                    ATTR_DOMAIN_POINT);
  faw.varray.set(0, smd->epsilon);

  faw = mesh->attributes_for_write().lookup_or_add_for_write<float>(levelset_tag,
                                                                    ATTR_DOMAIN_POINT);
  faw.varray.set(0, smd->levelset);

  faw = mesh->attributes_for_write().lookup_or_add_for_write<float>(bounds_tag, ATTR_DOMAIN_POINT);
  faw.varray.set(0, smd->boundingbox_expansion);

  faw = mesh->attributes_for_write().lookup_or_add_for_write<float>(formula_tag,
                                                                    ATTR_DOMAIN_POINT);
  faw.varray.set(0, static_cast<float>(smd->solid_angle_formula));

  return mesh;
}

static void panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *layout = panel->layout;

  PointerRNA ob_ptr;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, &ob_ptr);

  uiLayoutSetPropSep(layout, true);

  uiItemR(layout, ptr, "epsilon", UI_ITEM_NONE, IFACE_("Epsilon"), ICON_NONE);
  uiItemR(layout, ptr, "levelset", UI_ITEM_NONE, IFACE_("Level Set"), ICON_NONE);
  uiItemR(layout,
          ptr,
          "boundingbox_expansion",
          UI_ITEM_NONE,
          IFACE_("Bounding Box Expansion"),
          ICON_NONE);

  uiItemR(layout, ptr, "solid_angle_formula", UI_ITEM_NONE, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "use_grad_termination", UI_ITEM_NONE, nullptr, ICON_NONE);

  modifier_panel_end(layout, ptr);
}

static void panel_register(ARegionType *region_type)
{
  PanelType *panel_type = modifier_panel_register(region_type, eModifierType_Harnack, panel_draw);
}

ModifierTypeInfo modifierType_Harnack = {
    /*idname*/ "Harnack",
    /*name*/ N_("Harnack"),
    /*struct_name*/ "HarnackModifierData",
    /*struct_size*/ sizeof(HarnackModifierData),
    /*srna*/ &RNA_HarnackModifier,
    /*type*/ eModifierTypeType_Constructive,
    /*flags*/ eModifierTypeFlag_AcceptsMesh,
    /*icon*/ ICON_MOD_CURVE,

    /*copy_data*/ BKE_modifier_copydata_generic,  // save changes to data

    /*deform_verts*/ nullptr,
    /*deform_matrices*/ nullptr,
    /*deform_verts_EM*/ nullptr,
    /*deform_matrices_EM*/ nullptr,
    /*modify_mesh*/ harnack_applyModifier,
    /*modify_geometry_set*/ nullptr,

    /*init_data*/ init_data,
    /*required_data_mask*/ nullptr,
    /*free_data*/ nullptr,
    /*is_disabled*/ nullptr,
    /*update_depsgraph*/ nullptr,
    /*depends_on_time*/ nullptr,
    /*depends_on_normals*/ nullptr,
    /*foreach_ID_link*/ nullptr,
    /*foreach_tex_link*/ nullptr,
    /*free_runtime_data*/ nullptr,
    /*panel_register*/ panel_register,
    /*blend_write*/ nullptr,
    /*blend_read*/ nullptr,
};
