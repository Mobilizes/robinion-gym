# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild ``assets/robonionv2.usd`` from the robinion URDF shipped in ``robinion_description``.

The environments spawn the flattened crate ``assets/robonionv2.usd``; the URDF stays the only source
of truth for the mechanics. The asset is generated in three steps:

1. ``urdf_usd_converter`` (the importer that produced the committed asset) converts
   ``urdf/robinion2.urdf`` into the layered asset ``assets/robonionv2/robinion.usda`` with its
   generated payload under ``assets/robonionv2/Payload``.
2. The handful of things the URDF cannot express are authored on the asset interface layer: the
   PhysX articulation root, self-collision filtering, the four parallelogram loops that tie the
   back thigh/shin bars to the knee/ankle links, and the flat box foot sole that replaces the
   narrow convex-hull contact of the imported foot mesh.
3. The composed asset is flattened into the single file the articulation spawner reads.

The importer ships with Isaac Sim, so the project environment needs the ``isaacsim`` extra; no Isaac Sim
app has to be launched::

    uv run --extra isaacsim python scripts/prepare_asset.py

Pass ``--description-dir`` when the ``robinion_description`` checkout is not next to this repository.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import usdex.core
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from urdf_usd_converter import Converter

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTION_DIR = PROJECT_DIR.parent / "robinion_description"

# Name of the asset pair: the flattened ``assets/<stem>.usd`` and the layered ``assets/<stem>/`` directory.
ASSET_STEM = "robonionv2"

#: PhysX articulation root API; the importer only applies the backend-neutral and Newton ones.
PHYSX_ARTICULATION_API = "PhysxArticulationAPI"

#: Self-collision is filtered through one group that contains every collision shape of the robot.
SELF_COLLISION_GROUP = "RobinionSelfCollisionGroup"

#: Foot collision shape. The URDF/imported foot collision mesh is a convex hull of a thin, rounded
#: sole, so it touches the ground on a narrow strip. A flat box under the same link approximates the
#: rubber pad on the real robot and gives a stable support polygon.
FOOT_LINK_NAMES = ("right_foot_roll_link", "left_foot_roll_link")
FOOT_COLLISION_MESH_NAME = "foot_visual_1"
FOOT_SOLE_NAME = "foot_sole_collision"
FOOT_SOLE_COVERAGE = 0.9
"""Fraction of the foot mesh's footprint covered by the flat sole box."""
FOOT_SOLE_THICKNESS = 0.006
"""Sole box thickness [m]."""

#: The parallelogram bars are plain revolute joints in the URDF, so the linkage is an open chain and
#: the bars only follow the driven links because the environment actuates them together. Closing the
#: loop with spherical joints is what the real linkage does. Each entry is the loop joint name, the
#: bar it anchors, the driven link the bar attaches to, and the anchor point in both link frames.
#: The anchors come from the URDF's thigh/shin offsets: the bars are 0.2 m long, and the knee/ankle
#: frames are offset laterally by 0.04125 m from the bar mounting point.
LOOP_JOINTS = (
    (
        "right_back_thigh_loop_joint",
        "right_back_thigh_pitch_link",
        "right_knee_pitch_link",
        (0.0, 0.0, -0.2),
        (-0.04125, 0.0, 0.0),
    ),
    (
        "right_back_shin_loop_joint",
        "right_back_shin_pitch_link",
        "right_ankle_roll_pitch_link",
        (0.0, 0.0, -0.2),
        (-0.04125, 0.0, 0.0),
    ),
    (
        "left_back_thigh_loop_joint",
        "left_back_thigh_pitch_link",
        "left_knee_pitch_link",
        (0.0, 0.0, -0.2),
        (-0.04125, 0.0, 0.0),
    ),
    (
        "left_back_shin_loop_joint",
        "left_back_shin_pitch_link",
        "left_ankle_roll_pitch_link",
        (0.0, 0.0, -0.2),
        (-0.04125, 0.0, 0.0),
    ),
)


def convert(urdf_path: pathlib.Path, asset_dir: pathlib.Path) -> pathlib.Path:
    """Convert the URDF into the layered asset and return its interface layer."""
    converter = Converter(
        layer_structure=True,
        scene=True,
        ros_packages=[{"name": "robinion_description", "path": str(urdf_path.parent.parent)}],
    )
    return pathlib.Path(converter.convert(urdf_path, asset_dir).path)


def prepend_api(prim: Usd.Prim, api_schema: str) -> None:
    """Prepend an applied API schema without disturbing the ones the payload already applies.

    ``Usd.Prim.ApplyAPI`` requires the API's plugin to be registered, which is not the case for the
    PhysX schemas outside Isaac Sim; authoring the same ``prepend`` list op gives the same result.
    """
    existing = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    list_op = Sdf.TokenListOp()
    if existing.explicitItems:
        # An explicit list replaces prepended/appended items, so merge into it directly.
        list_op.explicitItems = [api_schema, *existing.explicitItems]
    else:
        list_op.prependedItems = [api_schema, *(existing.prependedItems or [])]
        list_op.appendedItems = existing.appendedItems
    prim.SetMetadata("apiSchemas", list_op)


def find_prim_by_name(stage: Usd.Stage, name: str) -> Usd.Prim:
    """Return the single prim named ``name``, which is how the loop joints locate their anchor links."""
    prims = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(prims) != 1:
        raise RuntimeError(f"expected exactly one prim named '{name}', found {len(prims)}")
    return prims[0]


def _mesh_bbox_in_link_frame(mesh_prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Return the local-space bounding box of a mesh after its own transform."""
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        raise RuntimeError(f"mesh '{mesh_prim.GetPath()}' has no points")
    transform = UsdGeom.Xformable(mesh_prim).GetLocalTransformation()
    transformed = [transform.Transform(Gf.Vec3d(point)) for point in points]
    lower = Gf.Vec3d(
        min(point[0] for point in transformed),
        min(point[1] for point in transformed),
        min(point[2] for point in transformed),
    )
    upper = Gf.Vec3d(
        max(point[0] for point in transformed),
        max(point[1] for point in transformed),
        max(point[2] for point in transformed),
    )
    return lower, upper


def author_foot_sole(stage: Usd.Stage, foot_link: Usd.Prim) -> None:
    """Replace the foot's convex-hull mesh contact with a flat box sole.

    The box covers most of the foot footprint and sits at the lowest point of the imported mesh, so
    it reproduces the flat rubber pad of the real foot instead of the narrow strip a convex hull of
    the rounded sole produces.
    """
    collision_mesh = None
    for child in foot_link.GetChildren():
        if child.GetName().endswith(FOOT_COLLISION_MESH_NAME):
            collision_mesh = child
            break
    if collision_mesh is None:
        raise RuntimeError(f"no '{FOOT_COLLISION_MESH_NAME}' mesh under '{foot_link.GetPath()}'")

    lower, upper = _mesh_bbox_in_link_frame(collision_mesh)
    length = FOOT_SOLE_COVERAGE * (upper[0] - lower[0])
    width = FOOT_SOLE_COVERAGE * (upper[1] - lower[1])
    center = Gf.Vec3d(
        0.5 * (lower[0] + upper[0]),
        0.5 * (lower[1] + upper[1]),
        lower[2] + 0.5 * FOOT_SOLE_THICKNESS,
    )

    sole = UsdGeom.Cube.Define(stage, foot_link.GetPath().AppendChild(FOOT_SOLE_NAME))
    sole.CreateSizeAttr().Set(1.0)
    sole_xform = UsdGeom.Xformable(sole)
    sole_xform.AddTranslateOp().Set(center)
    sole_xform.AddScaleOp().Set(Gf.Vec3f(length, width, FOOT_SOLE_THICKNESS))
    UsdGeom.Imageable(sole).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    sole_prim = sole.GetPrim()
    UsdPhysics.CollisionAPI.Apply(sole_prim)
    UsdPhysics.CollisionAPI(sole_prim).CreateCollisionEnabledAttr().Set(True)
    # Apply after the USD schema so ``UsdPhysics.CollisionAPI.Apply`` cannot replace the list op.
    prepend_api(sole_prim, "NewtonCollisionAPI")

    # Keep the mesh for rendering, but take it out of collision so only the flat box contacts.
    UsdPhysics.CollisionAPI.Apply(collision_mesh)
    UsdPhysics.CollisionAPI(collision_mesh).CreateCollisionEnabledAttr().Set(False)


def author_fixups(stage: Usd.Stage, foot_collision: str = "box") -> None:
    """Author everything the URDF cannot express on the asset interface layer."""
    root = stage.GetDefaultPrim()
    geometry = root.GetChild("Geometry")
    physics = root.GetChild("Physics")

    # The URDF root is an empty ``base_link`` that the robot root is welded to. The importer marks the
    # first link with mass as the articulation root, so the weld is dropped and ``lower_body_link``
    # becomes the root link the spawner expects (``fix_root_link=False``).
    base_link = geometry.GetChild("base_link")
    root_link = base_link.GetChild("lower_body_link")
    prepend_api(root_link, PHYSX_ARTICULATION_API)
    # ``newton:selfCollisionEnabled`` and ``physxArticulation:enabledSelfCollisions`` are the Newton and
    # PhysX counterparts of the same switch; the spawner turns self-collision back on for the scene.
    root_link.GetAttribute("newton:selfCollisionEnabled").Set(False)
    root_link.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, custom=True).Set(
        False
    )
    stage.GetPrimAtPath(physics.GetPath().AppendChild("base_to_lower_body_fixed_joint")).SetActive(False)

    # The imported foot collision is a convex hull of a rounded sole and only contacts on a narrow
    # strip. Replace it with a flat box sole for hardware-like support, or keep the mesh when asked.
    if foot_collision == "box":
        for foot_link_name in FOOT_LINK_NAMES:
            author_foot_sole(stage, find_prim_by_name(stage, foot_link_name))

    # Self-collision is enabled by the spawner, so filter it down to the pairs the URDF resolves: one
    # group holding every collision shape, with the group filtered against itself.
    collider_paths = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not collider_paths:
        raise RuntimeError("the converted asset has no collision shapes to filter")
    group = UsdPhysics.CollisionGroup.Define(stage, physics.GetPath().AppendChild(SELF_COLLISION_GROUP))
    group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(collider_paths)
    group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])
    group.CreateInvertFilteredGroupsAttr().Set(False)

    # The importer emits the physics scene outside the asset; opening the layered asset in a viewer
    # needs PhysX to treat it as one.
    scene = stage.GetPrimAtPath("/PhysicsScene")
    if scene.IsValid():
        prepend_api(scene, "PhysxSceneAPI")

    # Close the parallelogram linkage: the bars are unactuated in the real robot. The loop joints
    # are excluded from the articulation's degree-of-freedom count and hold the bars on the driven
    # links. The robot config keeps the bar revolute drives passive and out of the action space.
    for name, bar_link, driven_link, bar_anchor, driven_anchor in LOOP_JOINTS:
        joint = UsdPhysics.SphericalJoint.Define(stage, physics.GetPath().AppendChild(name))
        joint.CreateBody0Rel().SetTargets([find_prim_by_name(stage, bar_link).GetPath()])
        joint.CreateBody1Rel().SetTargets([find_prim_by_name(stage, driven_link).GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*bar_anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*driven_anchor))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        joint.CreateExcludeFromArticulationAttr().Set(True)


def flatten(stage: Usd.Stage, flattened_path: pathlib.Path) -> None:
    """Export the composed asset as the single crate file the spawner reads."""
    layer: Sdf.Layer = stage.Flatten()
    flattened_stage = Usd.Stage.Open(layer)
    # The interface layer carries viewer prims (camera, render settings, physics scene) that are not
    # part of the asset.
    default_prim_name = layer.defaultPrim
    for prim in flattened_stage.GetPseudoRoot().GetChildren():
        if prim.GetName() != default_prim_name:
            flattened_stage.RemovePrim(prim.GetPath())
    interface_path = pathlib.Path(stage.GetRootLayer().identifier).resolve()
    layer.customLayerData = {
        "creator": layer.customLayerData.get("creator", ""),
        "flattened_from": pathlib.Path(os.path.relpath(interface_path, PROJECT_DIR)).as_posix(),
    }
    # ``exportLayer`` authors a comment and provenance, and picks the file format from the extension.
    if not usdex.core.exportLayer(layer, str(flattened_path), layer.customLayerData["creator"]):
        raise RuntimeError(f"failed to export {flattened_path}")


def summarize(stage: Usd.Stage) -> None:
    """Print what the environment will see, so the conversion can be sanity-checked at a glance."""
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    revolute = [joint for joint in joints if joint.IsA(UsdPhysics.RevoluteJoint)]
    fixed = [joint for joint in joints if joint.IsA(UsdPhysics.FixedJoint)]
    loops = [joint for joint in joints if "loop_joint" in joint.GetName()]
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    masses = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.MassAPI)]
    total_mass = sum(prim.GetAttribute("physics:mass").Get() or 0.0 for prim in masses)
    print(f"[INFO]: {len(revolute)} revolute joints (the degrees of freedom), {len(fixed)} fixed, {len(loops)} loop")
    print(f"[INFO]: {len(bodies)} rigid bodies, {total_mass:.4f} kg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--description-dir",
        type=pathlib.Path,
        default=DEFAULT_DESCRIPTION_DIR,
        help="path to the robinion_description checkout (default: %(default)s)",
    )
    parser.add_argument("--assets-dir", type=pathlib.Path, default=PROJECT_DIR / "assets", help="output directory")
    parser.add_argument(
        "--foot-collision",
        choices=("box", "mesh"),
        default="box",
        help="foot collision shape: flat box sole (default) or the imported mesh convex hull",
    )
    args = parser.parse_args()

    urdf_path = args.description_dir / "urdf" / "robinion2.urdf"
    if not urdf_path.is_file():
        parser.error(f"no URDF at {urdf_path}; pass --description-dir pointing at the robinion_description checkout")

    asset_dir = args.assets_dir / ASSET_STEM
    interface_path = convert(urdf_path, asset_dir)
    print(f"[INFO]: converted {urdf_path} into {interface_path}")

    stage = Usd.Stage.Open(str(interface_path))
    author_fixups(stage, foot_collision=args.foot_collision)
    stage.Save()

    flattened_path = args.assets_dir / f"{ASSET_STEM}.usd"
    flatten(stage, flattened_path)
    print(f"[INFO]: flattened {interface_path} into {flattened_path}")
    summarize(Usd.Stage.Open(str(flattened_path)))


if __name__ == "__main__":
    main()
