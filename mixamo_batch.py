# mixamo_batch.py (Blender 5.0.1)
#
# Goal:
# - Input: 1 character FBX (with skin) + N animation FBXs (without skin or with skin)
# - Output:
#   - <character_name>.glb  (mesh + skeleton, no animations)
#   - <anim_name>.glb       (skeleton + that single animation; mesh optional)
#   - report.json           (compat checks + applied fixes)
#
# Usage:
# blender -b -P mixamo_batch.py -- \
#   --character ./input/y-bot.fbx \
#   --anims ./input/anims/idle.fbx ./input/anims/walk.fbx \
#   --out ./output \
#   --mode inplace \
#   --export-mesh-in-anims false \
#   --apply-scale true \
#   --fix-root true \
#   --limit-weights 4 \
#   --abort-on-incompatible false
#
import bpy
import os
import sys
import json
import re
from mathutils import Vector

try:
    from bpy_extras import anim_utils as _anim_utils
except Exception:
    _anim_utils = None


# -----------------------------
# Args parsing
# -----------------------------
def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    def pop_flag(name, default=None):
        if name not in argv:
            return default
        i = argv.index(name)
        if i + 1 >= len(argv):
            raise ValueError(f"Missing value for {name}")
        val = argv[i + 1]
        del argv[i : i + 2]
        return val

    def pop_bool(name, default=False):
        v = pop_flag(name, None)
        if v is None:
            return default
        return v.lower() in ("1", "true", "yes", "y", "on")

    def pop_int(name, default=None):
        v = pop_flag(name, None)
        if v is None:
            return default
        return int(v)

    def pop_list(name):
        if name not in argv:
            return []
        i = argv.index(name)
        del argv[i]
        items = []
        while argv and not argv[0].startswith("--"):
            items.append(argv.pop(0))
        return items

    args = {
        "character": pop_flag("--character", None),
        "anims": pop_list("--anims"),
        "out": pop_flag("--out", "./output"),
        "mode": pop_flag("--mode", "rootmotion").lower(),  # inplace | rootmotion
        "inplace_ground_lock": pop_bool("--inplace-ground-lock", True),
        "inplace_anchor": pop_flag("--inplace-anchor", "hips").lower(),  # hips | foot
        "export_mesh_in_anims": pop_bool("--export-mesh-in-anims", False),
        "apply_scale": pop_bool("--apply-scale", True),
        "fix_root": pop_bool("--fix-root", True),
        "limit_weights": pop_int("--limit-weights", 4),
        "abort_on_incompatible": pop_bool("--abort-on-incompatible", False),
    }

    if not args["character"]:
        raise ValueError("--character is required (FBX path).")
    if not args["anims"]:
        raise ValueError("--anims requires at least one FBX animation path.")
    if args["mode"] not in ("inplace", "rootmotion"):
        raise ValueError("--mode must be 'inplace' or 'rootmotion'.")
    if args["inplace_anchor"] not in ("hips", "foot"):
        raise ValueError("--inplace-anchor must be 'hips' or 'foot'.")

    return args


# -----------------------------
# Scene helpers
# -----------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def select_none():
    bpy.ops.object.select_all(action="DESELECT")


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Import/Export
# -----------------------------
def import_fbx(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    select_none()
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
    return list(bpy.context.selected_objects)


def export_glb_selected(path, export_anim=True, anim_name=None):
    # Blender GLTF exporter keywords change across versions.
    # Retry dropping unknown keywords to keep the script forward/backward compatible.
    kwargs = {
        "filepath": path,
        "export_format": "GLB",
        "export_apply": True,
        "export_yup": True,
        "export_animations": export_anim,
        "export_skins": True,
        "export_materials": "EXPORT",
        "export_morph": False,
        "use_selection": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_tangents": False,
        "export_colors": True,
        "export_cameras": False,
        "export_lights": False,
        "export_extras": False,
        "export_keep_originals": False,
        "export_image_format": "AUTO",
        "export_jpeg_quality": 90,
        "export_try_sparse_sk": False,
    }

    if export_anim:
        # Force a single clip from the currently assigned action.
        kwargs["export_animation_mode"] = "ACTIVE_ACTIONS"
        if anim_name:
            kwargs["export_nla_strips_merged_animation_name"] = anim_name

    while True:
        try:
            bpy.ops.export_scene.gltf(**kwargs)
            return
        except TypeError as ex:
            msg = str(ex)
            m = re.search(r'keyword "([^"]+)" unrecognized', msg)
            if not m:
                raise
            bad = m.group(1)
            if bad not in kwargs:
                raise
            print(f"[mixamo_batch] gltf exporter ignores unsupported arg: {bad}")
            del kwargs[bad]


# -----------------------------
# Object discovery
# -----------------------------
def find_armature(objs):
    for o in objs:
        if o.type == "ARMATURE":
            return o
    for o in bpy.context.scene.objects:
        if o.type == "ARMATURE":
            return o
    return None


def find_meshes_deformed_by_armature(arm_obj):
    meshes = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        for m in o.modifiers:
            if m.type == "ARMATURE" and m.object == arm_obj:
                meshes.append(o)
                break
    return meshes


def apply_object_transforms(obj, apply_loc=False, apply_rot=True, apply_scale=True):
    select_none()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(
        location=apply_loc,
        rotation=apply_rot,
        scale=apply_scale,
        properties=False,
    )


# -----------------------------
# Mixamo rig helpers
# -----------------------------
def strip_bone_prefixes(arm_obj):
    """
    Remove namespace/prefix like 'mixamorig:Hips' -> 'Hips'
    Avoid collisions: if new name already exists, keep original.
    Returns dict old->new for renamed bones.
    """
    arm = arm_obj.data
    ren = {}
    for b in arm.bones:
        if ":" in b.name:
            new_name = b.name.split(":")[-1]
            if new_name not in arm.bones:
                ren[b.name] = new_name

    if not ren:
        return {}

    select_none()
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    for old, new in ren.items():
        if old in eb:
            eb[old].name = new
    bpy.ops.object.mode_set(mode="OBJECT")
    return ren


def ensure_root_bone(arm_obj):
    """
    Create 'Root' bone above 'Hips' (or first top-level bone).
    Returns True if created, False if already existed.
    """
    arm = arm_obj.data
    if "Root" in arm.bones:
        return False

    select_none()
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones

    hips = eb.get("Hips")
    if hips is None:
        roots = [b for b in eb if b.parent is None]
        hips = roots[0] if roots else None

    root = eb.new("Root")
    if hips is not None:
        root.head = hips.head.copy()
        root.tail = hips.head + Vector((0.0, 0.0, 0.1))
        hips.parent = root
    else:
        root.head = Vector((0.0, 0.0, 0.0))
        root.tail = Vector((0.0, 0.0, 0.1))

    bpy.ops.object.mode_set(mode="OBJECT")
    return True


def limit_vertex_weights(mesh_obj, max_influences):
    if max_influences is None or max_influences <= 0:
        return False

    me = mesh_obj.data
    if not me.vertices:
        return False

    vgroups = mesh_obj.vertex_groups
    changed_any = False

    for v in me.vertices:
        groups = [(g.group, g.weight) for g in v.groups if g.weight > 0.0]
        if len(groups) <= max_influences:
            continue

        groups.sort(key=lambda x: x[1], reverse=True)
        keep = groups[:max_influences]
        drop = groups[max_influences:]

        for gi, _w in drop:
            try:
                vgroups[gi].remove([v.index])
                changed_any = True
            except Exception:
                pass

        s = sum(w for _gi, w in keep)
        if s > 0:
            for gi, w in keep:
                vgroups[gi].add([v.index], w / s, "REPLACE")
                changed_any = True

    return changed_any


def bone_name_set(arm_obj):
    return set(b.name for b in arm_obj.data.bones)


def check_anim_compat(base_arm, anim_arm):
    """
    Compatibility by bone-name set ratio.
    Returns (ok:bool, details:dict)
    """
    base = bone_name_set(base_arm)
    anim = bone_name_set(anim_arm)
    common = base.intersection(anim)

    details = {
        "base_bones": len(base),
        "anim_bones": len(anim),
        "common_bones": len(common),
        "common_ratio": (len(common) / max(1, len(base))) if base else 0.0,
        "status": "ok",
    }

    if not base or not anim:
        details["status"] = "incompatible"
        details["reason"] = "missing bones"
        return False, details

    ratio = details["common_ratio"]
    if ratio < 0.98:
        details["status"] = "incompatible"
        details["reason"] = "bone set mismatch"
        details["missing_in_anim"] = sorted(list(base - anim))[:30]
        details["extra_in_anim"] = sorted(list(anim - base))[:30]
        return False, details

    return True, details


def find_hips_bone_name(arm_obj):
    for b in arm_obj.data.bones:
        if b.name.lower().endswith("hips"):
            return b.name
    return None


def find_foot_bone_names(arm_obj):
    names = [b.name for b in arm_obj.data.bones]
    exact = {"leftfoot", "rightfoot", "lefttoebase", "righttoebase"}
    foots = []
    for n in names:
        ln = n.lower()
        if ln in exact or ln.endswith("foot") or ln.endswith("toe") or ln.endswith("toebase"):
            foots.append(n)
    return foots


def classify_bone_location_axes(arm_data_bone):
    """
    Determine which bone-local location axes map to world horizontal (XY)
    vs vertical (Z), based on the bone's rest orientation.
    Returns (horizontal_axes: list[int], vertical_axis: int).
    """
    mat = arm_data_bone.matrix_local
    best_vert = 1  # default: bone-local Y (common for upward-pointing bones)
    best_dot = 0.0
    for i in range(3):
        wz = abs(float(mat[2][i]))
        if wz > best_dot:
            best_dot = wz
            best_vert = i
    horizontal = sorted(i for i in range(3) if i != best_vert)
    return horizontal, best_vert


def object_uniform_scale(obj):
    s = obj.scale
    vals = [abs(float(s.x)), abs(float(s.y)), abs(float(s.z))]
    return sum(vals) / 3.0


def iter_action_fcurves(action):
    """
    Yield fcurves from both legacy Action API (action.fcurves)
    and newer slotted/layered Action API used by newer Blender versions.
    """
    seen = set()

    def push_many(collection):
        if collection is None:
            return
        try:
            items = list(collection)
        except Exception:
            return
        for fc in items:
            key = fc.as_pointer() if hasattr(fc, "as_pointer") else id(fc)
            if key in seen:
                continue
            seen.add(key)
            yield fc

    # Legacy API
    for fc in push_many(getattr(action, "fcurves", None)):
        yield fc

    # New API (layers/strips/channelbags)
    layers = getattr(action, "layers", None)
    if layers is None:
        return

    slots = []
    try:
        slots = list(getattr(action, "slots", []))
    except Exception:
        slots = []

    for layer in layers:
        strips = getattr(layer, "strips", None)
        if strips is None:
            continue
        for strip in strips:
            bags = getattr(strip, "channelbags", None)
            if bags is None:
                continue

            # Some builds expose channelbags as an iterable collection.
            try:
                iter_bags = list(bags)
            except Exception:
                iter_bags = []
            for bag in iter_bags:
                for fc in push_many(getattr(bag, "fcurves", None)):
                    yield fc

            # Other builds require resolving per-slot bag.
            if hasattr(bags, "slot"):
                for slot in slots:
                    try:
                        bag = bags.slot(slot)
                    except Exception:
                        continue
                    for fc in push_many(getattr(bag, "fcurves", None)):
                        yield fc


def get_or_create_action_fcurve_collection(action, target_obj=None):
    """
    Return a writable fcurve collection for both legacy actions and slotted actions.
    Returns (fcurves_collection, slot_or_none).
    """
    # Legacy API
    if hasattr(action, "fcurves"):
        return action.fcurves, None

    # Blender 4+/5+ slotted API
    slot = None
    slots = list(getattr(action, "slots", []))
    if target_obj is not None:
        for s in slots:
            try:
                if s.target_id_type in ("UNSPECIFIED", target_obj.id_type):
                    slot = s
                    break
            except Exception:
                continue
        if slot is None:
            slot = action.slots.new(target_obj.id_type, "Slot")
    elif slots:
        slot = slots[0]

    if slot is None:
        raise RuntimeError("Unable to create/read action slot for slotted Action API.")

    if _anim_utils is None:
        raise RuntimeError("bpy_extras.anim_utils unavailable for slotted Action API.")

    channelbag = _anim_utils.action_ensure_channelbag_for_slot(action, slot)
    return channelbag.fcurves, slot


def get_action_from_armature(arm_obj):
    """
    Works for Mixamo FBX without skin where action may land in NLA or be "loose".
    """
    ad = arm_obj.animation_data
    if ad:
        if ad.action:
            return ad.action
        for tr in ad.nla_tracks:
            for st in tr.strips:
                if st.action:
                    return st.action

    # Fallback: pick latest action with pose.bones curves
    for act in reversed(bpy.data.actions):
        if any(fc.data_path.startswith('pose.bones["') for fc in iter_action_fcurves(act)):
            return act
    return None


def bake_action_to_base(base_arm, src_arm, src_action, new_name):
    """
    Copy source action F-Curves into a fresh action on base armature.
    This avoids Action-slot binding issues while preserving original keys.
    """
    if base_arm.animation_data is None:
        base_arm.animation_data_create()
    if src_arm.animation_data is None:
        src_arm.animation_data_create()

    dst = bpy.data.actions.new(name=new_name)
    base_arm.animation_data.action = dst

    dst_fcurves, dst_slot = get_or_create_action_fcurve_collection(dst, base_arm)
    if dst_slot is not None and hasattr(base_arm.animation_data, "action_slot"):
        try:
            base_arm.animation_data.action_slot = dst_slot
        except Exception:
            pass

    frame_start = int(round(src_action.frame_range[0]))
    frame_end = int(round(src_action.frame_range[1]))
    if frame_end < frame_start:
        frame_start, frame_end = frame_end, frame_start

    src_names = set(pb.name for pb in src_arm.pose.bones)
    base_names = set(pb.name for pb in base_arm.pose.bones)
    common = src_names.intersection(base_names)
    src_obj_scale = object_uniform_scale(src_arm)
    base_obj_scale = object_uniform_scale(base_arm)
    location_scale_factor = 1.0
    if base_obj_scale > 1e-8:
        location_scale_factor = src_obj_scale / base_obj_scale

    def bone_from_data_path(path):
        prefix = 'pose.bones["'
        if not path.startswith(prefix):
            return None
        rest = path[len(prefix):]
        idx = rest.find('"]')
        if idx < 0:
            return None
        return rest[:idx]

    curves_copied = 0
    curves_skipped_non_bone = 0
    location_curves_scaled = 0
    for src_fc in iter_action_fcurves(src_action):
        data_path = src_fc.data_path
        if not data_path.startswith('pose.bones["'):
            curves_skipped_non_bone += 1
            continue

        bone_name = bone_from_data_path(data_path)
        if bone_name is None or bone_name not in common:
            continue

        group_name = src_fc.group.name if src_fc.group else None
        try:
            dst_fc = dst_fcurves.new(data_path, index=src_fc.array_index, group_name=group_name)
        except TypeError:
            try:
                dst_fc = dst_fcurves.new(data_path=data_path, index=src_fc.array_index, action_group=group_name)
            except TypeError:
                dst_fc = dst_fcurves.new(data_path=data_path, index=src_fc.array_index)
        dst_fc.extrapolation = src_fc.extrapolation

        kps = src_fc.keyframe_points
        dst_fc.keyframe_points.add(len(kps))
        for i, kp in enumerate(kps):
            dkp = dst_fc.keyframe_points[i]
            dkp.co = kp.co.copy()
            dkp.handle_left = kp.handle_left.copy()
            dkp.handle_right = kp.handle_right.copy()
            dkp.interpolation = kp.interpolation
            dkp.easing = kp.easing
            dkp.back = kp.back
            dkp.amplitude = kp.amplitude
            dkp.period = kp.period
            dkp.handle_left_type = kp.handle_left_type
            dkp.handle_right_type = kp.handle_right_type

            # Mixamo FBX imports often have animation armature at scale 0.01.
            # Compensate location keys so copied action matches base armature scale.
            if data_path.endswith('"].location') and abs(location_scale_factor - 1.0) > 1e-8:
                dkp.co[1] *= location_scale_factor
                dkp.handle_left[1] *= location_scale_factor
                dkp.handle_right[1] *= location_scale_factor
        dst_fc.update()
        if data_path.endswith('"].location') and abs(location_scale_factor - 1.0) > 1e-8:
            location_curves_scaled += 1
        curves_copied += 1

    return dst, {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "bones_baked": len(common),
        "curves_copied": curves_copied,
        "curves_skipped_non_bone": curves_skipped_non_bone,
        "src_obj_scale": src_obj_scale,
        "base_obj_scale": base_obj_scale,
        "location_scale_factor": location_scale_factor,
        "location_curves_scaled": location_curves_scaled,
    }


def clear_nla_tracks(arm_obj):
    ad = arm_obj.animation_data
    if not ad:
        return
    tracks = ad.nla_tracks
    while len(tracks) > 0:
        tracks.remove(tracks[0])


def evaluate_foot_ground_z(base_arm, action, foot_bone_names):
    """
    Evaluate foot bones in world space via depsgraph to find the ground level.
    Returns the minimum Z found across sampled frames, or None on failure.
    """
    if not foot_bone_names:
        return None

    ad = base_arm.animation_data
    if ad is None:
        return None

    prev_action = ad.action
    ad.action = action

    scene = bpy.context.scene
    frame_start = int(round(action.frame_range[0]))
    frame_end = int(round(action.frame_range[1]))
    if frame_end <= frame_start:
        ad.action = prev_action
        return None

    n_samples = min(20, frame_end - frame_start + 1)
    step = max(1, (frame_end - frame_start) / max(1, n_samples - 1))

    min_z = None
    try:
        for i in range(n_samples):
            f = frame_start + int(round(i * step))
            f = min(f, frame_end)
            scene.frame_set(f)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_arm = base_arm.evaluated_get(depsgraph)
            for bn in foot_bone_names:
                pb = eval_arm.pose.bones.get(bn)
                if pb is None:
                    continue
                world_z = (eval_arm.matrix_world @ pb.head).z
                if min_z is None or world_z < min_z:
                    min_z = world_z
    except Exception:
        pass

    ad.action = prev_action
    return min_z


def make_inplace_on_action(base_arm, action, ground_lock=True, anchor="hips"):
    """
    In-place strategy:
    - Prefer removing global drift from Root and armature object curves.
    - If absent, fallback to Hips drift removal so walk/run won't move in world.
    - Optionally lock vertical baseline too (ground_lock).
    """
    result = {
        "status": "skipped",
        "bone": None,
        "strategy": "root_object_then_hips",
        "ground_lock": bool(ground_lock),
        "anchor": anchor,
        "anchor_bone": None,
    }

    if action is None:
        result["status"] = "no_action"
        return result

    hips = find_hips_bone_name(base_arm) or "Hips"
    result["bone"] = hips

    def set_curve_constant(fc, value):
        changed_local = False
        for kp in fc.keyframe_points:
            if abs(kp.co[1] - value) > 1e-8:
                kp.co[1] = value
                changed_local = True
        fc.update()
        return changed_local

    def lock_axes(curves, axes, value=0.0):
        changed_local = False
        touched_local = False
        for fc in curves:
            if fc.array_index not in axes:
                continue
            touched_local = True
            if set_curve_constant(fc, value):
                changed_local = True
        return touched_local, changed_local

    def detrend_curve(fc):
        pts = list(fc.keyframe_points)
        if len(pts) < 2:
            return False
        t0 = float(pts[0].co[0])
        t1 = float(pts[-1].co[0])
        if abs(t1 - t0) <= 1e-8:
            return False
        v0 = float(pts[0].co[1])
        v1 = float(pts[-1].co[1])
        m = (v1 - v0) / (t1 - t0)
        changed_local = False
        for kp in pts:
            t = float(kp.co[0])
            trend = v0 + m * (t - t0)
            hl_trend = v0 + m * (float(kp.handle_left[0]) - t0)
            hr_trend = v0 + m * (float(kp.handle_right[0]) - t0)
            if abs(trend) > 1e-8 or abs(hl_trend) > 1e-8 or abs(hr_trend) > 1e-8:
                kp.co[1] -= trend
                kp.handle_left[1] -= hl_trend
                kp.handle_right[1] -= hr_trend
                changed_local = True
        fc.update()
        return changed_local

    def shift_curve_to_zero_start(fc):
        pts = list(fc.keyframe_points)
        if not pts:
            return False
        base = float(pts[0].co[1])
        if abs(base) <= 1e-8:
            return False
        changed_local = False
        for kp in pts:
            kp.co[1] -= base
            kp.handle_left[1] -= base
            kp.handle_right[1] -= base
            changed_local = True
        fc.update()
        return changed_local

    def shift_curve_by(fc, offset):
        if abs(offset) <= 1e-8:
            return False
        for kp in fc.keyframe_points:
            kp.co[1] += offset
            kp.handle_left[1] += offset
            kp.handle_right[1] += offset
        fc.update()
        return True

    def curve_end_start_delta(fc):
        pts = list(fc.keyframe_points)
        if len(pts) < 2:
            return 0.0
        return float(pts[-1].co[1] - pts[0].co[1])

    def remove_residual_end_drift(fc):
        # After main correction, remove only remaining end-to-start drift.
        pts = list(fc.keyframe_points)
        if len(pts) < 2:
            return False
        t0 = float(pts[0].co[0])
        t1 = float(pts[-1].co[0])
        if abs(t1 - t0) <= 1e-8:
            return False
        delta = float(pts[-1].co[1] - pts[0].co[1])
        if abs(delta) <= 1e-8:
            return False
        changed_local = False
        for kp in pts:
            alpha = (float(kp.co[0]) - t0) / (t1 - t0)
            alpha_hl = (float(kp.handle_left[0]) - t0) / (t1 - t0)
            alpha_hr = (float(kp.handle_right[0]) - t0) / (t1 - t0)
            kp.co[1] -= delta * alpha
            kp.handle_left[1] -= delta * alpha_hl
            kp.handle_right[1] -= delta * alpha_hr
            changed_local = True
        fc.update()
        return changed_local

    def location_curve_map(curves):
        return {fc.array_index: fc for fc in curves}

    # Determine bone-local axis mapping for both Root and Hips.
    # Mixamo bones point upward, so bone-local Y is world Z (vertical),
    # while bone-local X and Z map to world X/Y (horizontal).
    hips_bone = base_arm.data.bones.get(hips)
    if hips_bone:
        hips_horiz, hips_vert = classify_bone_location_axes(hips_bone)
    else:
        hips_horiz, hips_vert = [0, 1], 2

    root_bone = base_arm.data.bones.get("Root")
    if root_bone:
        root_horiz, root_vert = classify_bone_location_axes(root_bone)
        root_lock_axes = tuple(root_horiz) + ((root_vert,) if ground_lock else ())
    else:
        root_lock_axes = (0, 1, 2) if ground_lock else (0, 1)

    changed = False
    touched = False

    # Root carries the intended global trajectory when present.
    root_target = 'pose.bones["Root"].location'
    root_curves = [fc for fc in iter_action_fcurves(action) if fc.data_path == root_target]
    t_root, c_root = lock_axes(root_curves, root_lock_axes, value=0.0)
    touched = touched or t_root
    changed = changed or c_root

    # Object curves are in world space — axes 0,1 = horizontal, 2 = vertical.
    obj_target = "location"
    obj_curves = [fc for fc in iter_action_fcurves(action) if fc.data_path == obj_target]
    obj_lock_axes = (0, 1, 2) if ground_lock else (0, 1)
    t_obj, c_obj = lock_axes(obj_curves, obj_lock_axes, value=0.0)
    touched = touched or t_obj
    changed = changed or c_obj

    hips_target = f'pose.bones["{hips}"].location'
    hips_curves = [fc for fc in iter_action_fcurves(action) if fc.data_path == hips_target]
    hips_by_axis = location_curve_map(hips_curves)
    should_fallback = (not touched)
    if not should_fallback and hips_curves:
        # Even when Root/Object exists, fallback if Hips still has net horizontal drift.
        max_drift = 0.0
        for ax in hips_horiz:
            fc = hips_by_axis.get(ax)
            if fc:
                max_drift = max(max_drift, abs(curve_end_start_delta(fc)))
        if max_drift > 1e-6:
            should_fallback = True

    if should_fallback:
        if not hips_curves:
            result["status"] = "no_root_object_or_hips_location_curves"
            return result

        # Remove horizontal drift from Hips via linear detrend.
        for axis in hips_horiz:
            hips_fc = hips_by_axis.get(axis)
            if hips_fc is None:
                continue
            if detrend_curve(hips_fc):
                changed = True
            if remove_residual_end_drift(hips_fc):
                changed = True

        if ground_lock and hips_vert in hips_by_axis:
            if anchor == "foot":
                foot_names = find_foot_bone_names(base_arm)
                ground_z = evaluate_foot_ground_z(base_arm, action, foot_names)
                if ground_z is not None:
                    result["anchor_bone"] = ",".join(foot_names[:4])
                    if shift_curve_by(hips_by_axis[hips_vert], -ground_z):
                        changed = True
                else:
                    if shift_curve_to_zero_start(hips_by_axis[hips_vert]):
                        changed = True
            else:
                if shift_curve_to_zero_start(hips_by_axis[hips_vert]):
                    changed = True
            # Remove any remaining vertical drift so the loop closes cleanly.
            if remove_residual_end_drift(hips_by_axis[hips_vert]):
                changed = True

        result["status"] = "applied_fallback" if changed else "already_inplace_fallback"
    else:
        result["status"] = "applied" if changed else "already_inplace"
    return result


# -----------------------------
# Selection for export
# -----------------------------
def select_for_character_export(arm_obj, mesh_objs):
    select_none()
    arm_obj.select_set(True)
    for m in mesh_objs:
        m.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj


def select_for_anim_export(arm_obj, mesh_objs, export_mesh):
    select_none()
    arm_obj.select_set(True)
    if export_mesh:
        for m in mesh_objs:
            m.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj


# -----------------------------
# Cleanup helpers
# -----------------------------
def delete_objects(objs):
    select_none()
    for o in objs:
        if o.name in bpy.context.scene.objects:
            o.select_set(True)
    bpy.ops.object.delete(use_global=False, confirm=False)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args(sys.argv)
    outdir = os.path.abspath(args["out"])
    ensure_outdir(outdir)

    character_path = os.path.abspath(args["character"])
    character_base = os.path.splitext(os.path.basename(character_path))[0]
    character_glb = os.path.join(outdir, f"{character_base}.glb")

    report = {
        "inputs": {
            "character": character_path,
            "anims": [os.path.abspath(a) for a in args["anims"]],
            "mode": args["mode"],
            "inplace_ground_lock": args["inplace_ground_lock"],
            "inplace_anchor": args["inplace_anchor"],
            "export_mesh_in_anims": args["export_mesh_in_anims"],
            "apply_scale": args["apply_scale"],
            "fix_root": args["fix_root"],
            "limit_weights": args["limit_weights"],
        },
        "character": {
            "armature": None,
            "meshes": [],
            "bone_rename": {},
            "root": {"created": False},
            "weights": {"limited_on_meshes": []},
            "export": {"path": character_glb},
        },
        "anims": [],
        "exports": [],
    }

    # ---- Load character FBX
    reset_scene()
    imported = import_fbx(character_path)
    base_arm = find_armature(imported)
    if base_arm is None:
        raise RuntimeError("No armature found importing character FBX.")

    base_meshes = find_meshes_deformed_by_armature(base_arm)
    report["character"]["armature"] = base_arm.name
    report["character"]["meshes"] = [m.name for m in base_meshes]

    # Normalize bone names (strip mixamorig:)
    ren = strip_bone_prefixes(base_arm)
    report["character"]["bone_rename"] = {
        "renamed": len(ren),
        "examples": list(ren.items())[:10],
    }

    # Apply transforms for stability
    if args["apply_scale"]:
        apply_object_transforms(base_arm, apply_loc=False, apply_rot=True, apply_scale=True)
        for m in base_meshes:
            apply_object_transforms(m, apply_loc=False, apply_rot=True, apply_scale=True)

    # Ensure Root bone
    if args["fix_root"]:
        created = ensure_root_bone(base_arm)
        report["character"]["root"]["created"] = created

    # Limit weights to N
    if args["limit_weights"] is not None:
        for m in base_meshes:
            if limit_vertex_weights(m, args["limit_weights"]):
                report["character"]["weights"]["limited_on_meshes"].append(m.name)

    # Export character GLB (no animations)
    select_for_character_export(base_arm, base_meshes)
    export_glb_selected(character_glb, export_anim=False)
    report["exports"].append({"type": "character", "path": character_glb})

    # ---- Process animations
    for anim_path in args["anims"]:
        anim_path = os.path.abspath(anim_path)
        anim_base = os.path.splitext(os.path.basename(anim_path))[0]
        anim_glb = os.path.join(outdir, f"{anim_base}.glb")

        entry = {
            "input": anim_path,
            "name": anim_base,
            "bone_rename": {},
            "compat": {},
            "bake": {},
            "inplace": {},
            "export": {"path": anim_glb, "mesh_included": args["export_mesh_in_anims"]},
        }

        # Import anim FBX into current scene (temp objects)
        before = set(bpy.context.scene.objects)
        imported_anim = import_fbx(anim_path)
        after = set(bpy.context.scene.objects)
        new_objs = list(after - before)

        anim_arm = find_armature(new_objs) or find_armature(imported_anim)
        if anim_arm is None:
            entry["compat"] = {"status": "incompatible", "reason": "no armature in anim FBX"}
            report["anims"].append(entry)
            if args["abort_on_incompatible"]:
                raise RuntimeError(f"Incompatible anim (no armature): {anim_path}")
            delete_objects(new_objs)
            continue

        # Strip bone prefixes on anim rig too
        ren2 = strip_bone_prefixes(anim_arm)
        entry["bone_rename"] = {"renamed": len(ren2), "examples": list(ren2.items())[:10]}

        ok, compat = check_anim_compat(base_arm, anim_arm)
        entry["compat"] = compat
        if not ok:
            report["anims"].append(entry)
            if args["abort_on_incompatible"]:
                raise RuntimeError(f"Incompatible anim: {anim_path}")
            delete_objects(new_objs)
            continue

        src_action = get_action_from_armature(anim_arm)
        if src_action is None:
            entry["compat"] = {"status": "incompatible", "reason": "no action found in anim FBX"}
            report["anims"].append(entry)
            if args["abort_on_incompatible"]:
                raise RuntimeError(f"No action found in anim: {anim_path}")
            delete_objects(new_objs)
            continue

        # Bake source animation onto base rig (single clean action for export).
        clear_nla_tracks(base_arm)
        dst_action, bake_info = bake_action_to_base(base_arm, anim_arm, src_action, anim_base)
        entry["bake"] = bake_info

        if args["mode"] == "inplace":
            entry["inplace"] = make_inplace_on_action(
                base_arm,
                dst_action,
                ground_lock=args["inplace_ground_lock"],
                anchor=args["inplace_anchor"],
            )
        else:
            entry["inplace"] = {"status": "preserved_rootmotion"}

        # Remove imported anim rig and source action before export.
        # This guarantees only the copied destination clip is left to export.
        delete_objects(new_objs)
        try:
            bpy.data.actions.remove(src_action)
        except Exception:
            pass

        # Export animation GLB (skeleton + animation; mesh optional)
        select_for_anim_export(base_arm, base_meshes, export_mesh=args["export_mesh_in_anims"])
        export_glb_selected(anim_glb, export_anim=True, anim_name=anim_base)

        report["exports"].append({"type": "anim", "name": anim_base, "path": anim_glb})
        report["anims"].append(entry)

        # Keep only transient action for this export; avoid accumulating clips.
        try:
            if base_arm.animation_data:
                base_arm.animation_data.action = None
            bpy.data.actions.remove(dst_action)
        except Exception:
            pass

    # Report file
    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[mixamo_batch] Done. Output: {outdir}")
    print(f"[mixamo_batch] Character GLB: {character_glb}")
    print(f"[mixamo_batch] Report: {report_path}")


if __name__ == "__main__":
    main()
