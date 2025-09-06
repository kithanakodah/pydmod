from aabbtree import AABB
from argparse import ArgumentParser
from dataclasses import astuple
from DbgPack import AssetManager
from io import BytesIO
from glob import glob
from pathlib import Path
from PIL import Image as PILImage
from pygltflib import *
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple
import warnings

import logging
import multiprocessing
import multiprocessing.pool
import numpy

# Suppress harmless overflow warnings from the Jenkins hashing algorithm
warnings.filterwarnings("ignore", "overflow encountered in scalar add")

from adr_converter import dme_from_adr
from cnk_loader import ForgelightChunk
from dme_converter import append_dme_to_gltf, save_textures
from dme_loader import DME, DMAT  # Add this import
from utils.gltf_helpers import add_chunk_to_gltf_simple
from zone_loader import Zone
from zone_loader.data_classes import LightType

logger = logging.getLogger("Zone Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))


def load_chunk_from_extracted(base_path: Path, chunk_name: str) -> bytes:
    """Load chunk data from extracted files"""
    chunk_path = base_path / chunk_name
    if chunk_path.exists():
        with open(chunk_path, 'rb') as f:
            return f.read()
    return None


# This function is not used for terrain but is kept for actors.
def save_chunk_textures(*args, **kwargs):
    pass


def get_gltf_rotation(rotation: Tuple[float, float, float]):
    r = Rotation.from_euler("yzx", rotation, False)
    rot = list(r.as_quat())
    rot[1] *= -1
    rot[3] *= -1
    temp = rot[0] * -1
    rot[0] = rot[2] * -1
    rot[2] = temp
    return rot


def dme_from_adr_extracted(asset_dir: Path, adr_file: str):
    """Load DME from extracted assets directory instead of pack files"""
    from io import StringIO
    from xml.etree import ElementTree as ET
    from dme_loader import DME, DMAT

    logger.info(f"Loading ADR file {adr_file} from extracted assets...")

    # Load ADR file from extracted assets
    adr_path = asset_dir / adr_file
    if not adr_path.exists():
        logger.error(f"ADR file not found: {adr_path}")
        return None

    try:
        with open(adr_path, 'r') as file:
            tree = ET.parse(file)
            root = tree.getroot()

        if root.tag != "ActorRuntime":
            logger.error("File's root XML tag was not ActorRuntime!")
            return None

        # Get base model info
        base = root.find("Base")
        if base is None:
            logger.error("No base model present in Actor Runtime file")
            return None

        dme_name = base.get("fileName")
        dma_palette = base.get("paletteName")

        # Load DME from extracted assets
        dme_path = asset_dir / dme_name
        if not dme_path.exists():
            logger.error(f"DME file not found: {dme_path}")
            return None

        with open(dme_path, 'rb') as dme_file:
            dme = DME.load(dme_file, adr_file=adr_file, dme_file=dme_name, asset_dir=asset_dir)

        dme.name = dme_name

        # Handle palette/material files if different
        if Path(dma_palette).stem.lower() != Path(dme_name).stem.lower():
            logger.info(f"Loading palette with different name: {Path(dma_palette).stem} vs {Path(dme_name).stem}")

            dma_path = asset_dir / dma_palette
            if dma_path.exists():
                logger.info(f"Loading DMA from extracted assets: {dma_path}")
                with open(dma_path, 'rb') as dma_file:
                    dma_data = dma_file.read()
                    dma_file_io = BytesIO(dma_data)
                    logger.info(f"Loaded palette asset with length {len(dma_data)}, creating DMAT...")
                    dmat = DMAT.load(dma_file_io, len(dma_data))
                    dme.dmat = dmat

        return dme

    except Exception as e:
        logger.error(f"Error loading {adr_file}: {e}")
        return None


def append_dme_to_gltf_extracted(gltf: GLTF2, dme: DME, asset_dir: Path, mats: Dict[int, List[int]],
                                 textures: Dict[str, PILImage.Image], image_indices: Dict[str, int], offset: int,
                                 blob: bytes,
                                 dme_name: str, include_skeleton: bool = True) -> Tuple[int, bytes]:
    """Modified version of append_dme_to_gltf that uses extracted assets instead of AssetManager"""
    from utils.gltf_helpers import texture_name_to_indices
    import re
    from re import RegexFlag

    sampler = 0
    if len(gltf.samplers) == 0 or gltf.samplers[sampler].wrapS != REPEAT:
        sampler = len(gltf.samplers)
        gltf.samplers.append(Sampler(magFilter=LINEAR, minFilter=LINEAR))
    texture_groups_dict = {}
    atlas_set = set()
    for texture in dme.dmat.textures:
        group_match = re.match(r"(.*)_(C|c|N|n|S|s)\.dds", texture)
        atlas_match = re.match(r".*(atlas|CommandControlTerminal).*\.dds", texture, flags=RegexFlag.IGNORECASE)
        if not group_match and atlas_match and texture not in atlas_set:
            atlas_set.add(texture)
        if not group_match:
            continue
        if group_match.group(1) not in texture_groups_dict:
            texture_groups_dict[group_match.group(1)] = 0
        texture_groups_dict[group_match.group(1)] += 1

    texture_groups = [name for name, _ in
                      sorted(list(texture_groups_dict.items()), key=lambda pair: pair[1], reverse=True)]

    logger.info(f"Texture groups: {texture_groups}")
    logger.info(f"Atlas textures: {list(atlas_set)}")

    mat_info = []
    SUFFIX_TO_TYPE = {
        "_C.png": "base",
        "_MR.png": "met",
        "_N.png": "norm",
        "_E.png": "emis"
    }

    for name in texture_groups:
        for suffix in ["_C.dds", "_N.dds", "_S.dds"]:
            if str(Path(name + suffix).with_suffix(".png")) not in textures:
                load_texture_extracted(asset_dir, gltf, textures, name + suffix, image_indices, sampler)
        mat_info_entry = {"base": None, "met": None, "norm": None, "emis": None}
        for suffix in ["_C.png", "_MR.png", "_N.png", "_E.png"]:
            if (name + suffix) not in image_indices:
                continue

            if (name + suffix) in texture_name_to_indices:
                if suffix == "_N.png":
                    mat_info_entry["norm"] = NormalMaterialTexture(index=texture_name_to_indices[name + suffix])
                else:
                    mat_info_entry[SUFFIX_TO_TYPE[suffix]] = TextureInfo(index=texture_name_to_indices[name + suffix])
                continue

            if suffix == "_N.png":
                mat_info_entry["norm"] = NormalMaterialTexture(index=image_indices[name + suffix])
            else:
                mat_info_entry[SUFFIX_TO_TYPE[suffix]] = TextureInfo(index=image_indices[name + suffix])

            texture_name_to_indices[name + suffix] = image_indices[name + suffix]

        mat_info.append(mat_info_entry)

    atlas_texture = None
    for atlas in atlas_set:
        if str(Path(atlas).with_suffix(".png")) in texture_name_to_indices:
            atlas_texture = texture_name_to_indices[str(Path(atlas).with_suffix(".png"))]
            continue
        atlas_texture = len(gltf.textures)
        load_texture_extracted(asset_dir, gltf, textures, atlas, image_indices, sampler)

    mesh_materials = []
    assert len(dme.meshes) == len(dme.dmat.materials), "Mesh count != material count"

    for i, material in enumerate(dme.dmat.materials):
        if material.namehash not in mats:
            mats[material.namehash] = []

        if i < len(mat_info):
            mat_textures: Dict[str, Optional[TextureInfo]] = mat_info[i]
        elif material.name() == 'BumpRigidHologram2SidedBlend':
            mat_textures: Dict[str, Optional[TextureInfo]] = {"base": None, "met": None, "norm": None,
                                                              "emis": TextureInfo(index=atlas_texture)}
        else:
            mat_textures: Dict[str, Optional[TextureInfo]] = {"base": None, "met": None, "norm": None, "emis": None}

        # look for existing material that uses same textures
        for mat_index in mats[material.namehash]:
            logger.debug(f"gltf.materials[{mat_index}] == {gltf.materials[mat_index]}")
            logger.debug(f"mat_textures == {mat_textures}")
            baseColorTexture = gltf.materials[mat_index].pbrMetallicRoughness.baseColorTexture
            emissiveTexture = gltf.materials[mat_index].emissiveTexture
            if baseColorTexture is not None and mat_textures["base"] is not None and baseColorTexture.index == \
                    mat_textures["base"].index:
                logger.info("Found existing material with same base texture")
                mesh_materials.append(mat_index)
                break
            elif emissiveTexture is not None and mat_textures["emis"] is not None and emissiveTexture.index == \
                    mat_textures["emis"].index:
                logger.info("Found existing material with same emissive texture")
                mesh_materials.append(mat_index)
                break
            elif baseColorTexture is None and mat_textures["base"] is None:
                logger.info("Found existing material with same (null) base texture")
                mesh_materials.append(mat_index)
                break

        # material was found and assigned to this mesh, continue
        if len(mesh_materials) > i:
            continue

        # material was not found - create new
        logger.info(f"Creating new material instance #{len(mats[material.namehash]) + 1}")
        mats[material.namehash].append(len(gltf.materials))
        mesh_materials.append(len(gltf.materials))

        EMISSIVE_STRENGTH = 5
        new_mat = Material(
            name=material.name(),
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=mat_textures["base"],
                metallicRoughnessTexture=mat_textures["met"],
                baseColorFactor=[1, 1, 1, 1] if mat_textures["base"] is not None else [0, 0, 0, 1]
            ),
            normalTexture=mat_textures["norm"],
            emissiveTexture=mat_textures["emis"],
            emissiveFactor=[EMISSIVE_STRENGTH if mat_textures["emis"] is not None else 0] * 3,
            alphaCutoff=None,
            alphaMode=OPAQUE if material.name() != "Foliage" else BLEND
        )
        gltf.materials.append(new_mat)

    from utils.gltf_helpers import add_mesh_to_gltf, add_skeleton_to_gltf

    for i, mesh in enumerate(dme.meshes):
        logger.info(f"Writing mesh {i + 1} of {len(dme.meshes)}")
        material_index = mesh_materials[i]
        swapped = False
        if len(dme.bone_map2) > 0 and i == 1:
            logger.warning(
                "Swapping around bone maps since there were bones with the same index in the dme bone map entries.")
            logger.warning("Theoretically this should only happen for high bone count models (Colossus is one)")
            swapped = True
            temp = dme.bone_map
            dme.bone_map = dme.bone_map2
            dme.bone_map2 = temp
        offset, blob = add_mesh_to_gltf(gltf, dme, mesh, material_index, offset, blob)
        if swapped:
            dme.bone_map2 = dme.bone_map
            dme.bone_map = temp

    if len(dme.bones) > 0 and include_skeleton:
        offset, blob = add_skeleton_to_gltf(gltf, dme, offset, blob)

    return offset, blob


def load_texture_extracted(asset_dir: Path, gltf: GLTF2, textures: Dict[str, PILImage.Image], name: str,
                           texture_indices: Dict[str, int], sampler: int = 0):
    """Load texture from extracted assets directory instead of AssetManager"""
    import re
    from utils.gltf_helpers import unpack_normal
    import os

    # Try loading from extracted assets
    texture_path = asset_dir / name
    if not texture_path.exists():
        logger.warning(f"Could not find {name} in extracted assets, skipping...")
        return

    logger.info(f"Loaded {name}")

    with open(texture_path, 'rb') as f:
        texture_data = f.read()

    im = PILImage.open(BytesIO(texture_data))
    if re.match(".*_(s|S).dds", name):
        unpack_specular_extracted(asset_dir, gltf, textures, im, name, texture_indices, sampler)
        return
    elif re.match(".*_(n|N).dds", name):
        unpack_normal(gltf, textures, im, name, texture_indices, sampler)
        return
    elif re.match(".*_(c|C).dds", name):
        texture_indices[str(Path(name).with_suffix(".png"))] = len(gltf.images)
        name = str(Path(name).with_suffix(".png"))
        textures[name] = im
    else:
        texture_indices[str(Path(name).with_suffix(".png"))] = len(gltf.images)
        name = str(Path(name).with_suffix(".png"))
        textures[name] = im
    gltf.textures.append(Texture(source=len(gltf.images), sampler=sampler, name=name))
    gltf.images.append(Image(uri="textures" + os.sep + name))


def unpack_specular_extracted(asset_dir: Path, gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image,
                              name: str, texture_indices: Dict[str, int], sampler: int = 0):
    """Modified version of unpack_specular that uses extracted assets"""
    import os
    from PIL import ImageChops

    metallic = im.getchannel("R")
    roughness = im.getchannel("A")
    metallicRoughness = PILImage.merge("RGB", [metallic, roughness, metallic])
    albedoName = name[:-5] + "C.dds" if name[-5] == "S" else "c.dds"

    # Try extracted assets for albedo texture
    albedo_path = asset_dir / albedoName
    if albedo_path.exists():
        with open(albedo_path, 'rb') as f:
            albedo_data = f.read()
        albedo = PILImage.open(BytesIO(albedo_data))
        albedoRGB = albedo.convert(mode="RGB")
        constant = PILImage.new(mode="RGB", size=albedo.size)
        mask = ImageChops.multiply(
            PILImage.eval(im.getchannel("B").resize(constant.size), lambda x: 255 if x > 50 else 0),
            albedo.getchannel("A"))
        emissive = PILImage.composite(albedoRGB, constant, mask)
        albedo.close()
        constant.close()
    else:
        logger.warning(f"Failed to load albedo {albedoName} from extracted assets, using fallback")
        emissive = im.getchannel("B").convert(mode="RGB")

    ename = name[:-5] + "E.png"
    textures[ename] = emissive
    texture_indices[ename] = len(gltf.textures)
    gltf.textures.append(Texture(source=len(gltf.images), sampler=sampler, name=ename))
    gltf.images.append(Image(uri="textures" + os.sep + ename))
    mrname = name[:-5] + "MR.png"
    textures[mrname] = metallicRoughness
    texture_indices[mrname] = len(gltf.textures)
    gltf.textures.append(Texture(source=len(gltf.images), sampler=sampler, name=mrname))
    gltf.images.append(Image(uri="textures" + os.sep + mrname))


def main():
    parser = ArgumentParser(description="A utility to convert Zone files to GLTF2 files")
    parser.add_argument("input_file", type=str,
                        help="Path of the input Zone file, either already extracted or from game assets")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"],
                        help="The output format to use, required for conversion")
    parser.add_argument("--asset-dir", type=str, default="M:/H1Z1_assets",
                        help="Path to the extracted H1Z1 assets directory (default: M:/H1Z1_assets)")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count",
                        default=0)
    parser.add_argument("--skip-textures", "-s", help="Skips saving textures", action="store_true")
    parser.add_argument("--terrain-enabled", "-t", help="Load terrain chunks as models into the result",
                        action="store_true")
    parser.add_argument("--actors-enabled", "-a", help="Loads static actor files as models (buildings, trees, etc)",
                        action="store_true")
    parser.add_argument("--lights-enabled", "-i", help="Adds lights to the output file", action="store_true")
    parser.add_argument("--live", "-l", help="Loads live assets rather than test", action="store_true")
    parser.add_argument("--bounding-box", "-b",
                        help="The x1 z1 x2 z2 bounding box of the zone that should be loaded. Loads any object with a bounding box that intersects",
                        nargs=4, type=float)
    args = parser.parse_args()

    bounding_box = None
    if args.bounding_box is not None:
        bounding_box = AABB(
            ((args.bounding_box[0], args.bounding_box[2]), (0.0, 1000.0), (args.bounding_box[1], args.bounding_box[3])))
        logger.info(f"Using bounding box {bounding_box}")

    if not (args.terrain_enabled or args.actors_enabled or args.lights_enabled):
        parser.error("No model/light loading enabled! Use either/all of -a, -t, or -i to load models and/or lights")

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    # Convert asset-dir to Path object for consistent usage
    asset_dir = Path(args.asset_dir)
    logger.info(f"Using asset directory: {asset_dir}")

    # Create a dummy manager since we're not using pack files
    manager = None

    # Load zone file from asset directory
    zone_path = asset_dir / args.input_file
    try:
        file = open(zone_path, "rb")
        logger.info(f"Loading zone file from asset directory: {zone_path}")
    except FileNotFoundError:
        logger.error(f"Zone file not found in asset directory: {zone_path}")
        return -1

    zone = Zone.load(file)

    gltf = GLTF2()
    mats: Dict[int, Material] = {}
    textures: Dict[str, PILImage.Image] = {}
    blob: bytes = b''
    offset = 0
    instance_nodes = []
    chunk_nodes = []
    terrain_parent = None
    actor_parent = None
    light_parent = None

    if args.terrain_enabled:
        logger.info(f"Terrain enabled with bounding box: {args.bounding_box if args.bounding_box else 'None'}")

        # H1Z1 coordinate system debug info
        logger.info(f"Zone header: start_x={zone.header.start_x}, start_y={zone.header.start_y}")
        logger.info(f"Zone size: chunks_x={zone.header.chunks_x}, chunks_y={zone.header.chunks_y}")
        logger.info(
            f"Chunk coordinates range: X=[{zone.header.start_x} to {zone.header.start_x + zone.header.chunks_x - 4}], Y=[{zone.header.start_y} to {zone.header.start_y + zone.header.chunks_y - 4}]")

        chunks_loaded = 0
        chunks_checked = 0

        # OPTIMIZED: Calculate which chunks we actually need to check
        if bounding_box is not None:
            bbox_x_min, bbox_x_max = bounding_box.limits[0]
            bbox_z_min, bbox_z_max = bounding_box.limits[2]

            # Convert bounding box to chunk coordinates
            # Each chunk covers 256 world units, starting from chunk coord * 64
            chunk_start_x = int((bbox_x_min - 128) // 64) & ~3  # Round down to multiple of 4
            chunk_end_x = int((bbox_x_max + 128) // 64) | 3  # Round up to multiple of 4
            chunk_start_z = int((bbox_z_min - 128) // 64) & ~3  # Round down to multiple of 4
            chunk_end_z = int((bbox_z_max + 128) // 64) | 3  # Round up to multiple of 4

            # Clamp to zone bounds
            chunk_start_x = max(chunk_start_x, zone.header.start_x)
            chunk_end_x = min(chunk_end_x, zone.header.start_x + zone.header.chunks_x - 4)
            chunk_start_z = max(chunk_start_z, zone.header.start_y)
            chunk_end_z = min(chunk_end_z, zone.header.start_y + zone.header.chunks_y - 4)

            logger.info(
                f"Optimized chunk range: X=[{chunk_start_x} to {chunk_end_x}], Z=[{chunk_start_z} to {chunk_end_z}]")
        else:
            # No bounding box - check all chunks
            chunk_start_x = zone.header.start_x
            chunk_end_x = zone.header.start_x + zone.header.chunks_x - 4
            chunk_start_z = zone.header.start_y
            chunk_end_z = zone.header.start_y + zone.header.chunks_y - 4

        for x in range(chunk_start_x, chunk_end_x + 1, 4):
            for y in range(chunk_start_z, chunk_end_z + 1, 4):
                chunks_checked += 1

                # FIXED: Chunk positioning is now handled by calculate_verts_js_exact
                # Just use chunk coordinates directly
                chunk_coord_x = x
                chunk_coord_z = y

                # SIMPLIFIED: Skip bounding box check for now (let all chunks load)
                # You can re-implement this later using the chunk's actual AABB after loading
                if bounding_box is not None:
                    # For now, just load all chunks to test the gap fix
                    overlaps = True

                    # DEBUG: Log first few chunks only
                    if chunks_checked <= 5:
                        logger.info(
                            f"DEBUG Chunk ({x},{y}): coord=({chunk_coord_x},{chunk_coord_z}) loading={overlaps}")

                    if not overlaps:
                        continue

                # Use the configurable asset directory
                chunk_name = f"{Path(args.input_file).stem}_{x}_{y}.cnk1"

                logger.info(f"Loading chunk: {chunk_name}")

                # Load the decompressed chunk geometry data
                chunk_data_bytes = load_chunk_from_extracted(asset_dir, chunk_name)

                if chunk_data_bytes is None:
                    logger.warning(f"Could not find file: {chunk_name}")
                    continue

                chunk = ForgelightChunk.load(BytesIO(chunk_data_bytes), compressed=False)
                logger.info(f"Loading chunk {chunk_name} with coordinates ({x},{y})")

                # FIXED: Use the new method that handles positioning internally
                chunk.calculate_verts_js_exact(chunk_x_grid=x, chunk_z_grid=y)

                material_index = None
                node_start = len(gltf.nodes)
                offset, blob = add_chunk_to_gltf_simple(gltf, chunk, material_index, offset, blob)
                chunk_nodes.append(len(gltf.nodes))

                gltf.nodes.append(Node(
                    name=Path(chunk_name).stem,
                    children=list(range(node_start, len(gltf.nodes)))
                ))

                chunks_loaded += 1

        logger.info(f"Checked {chunks_checked} chunks, loaded {chunks_loaded} terrain chunks")

        if chunks_loaded > 0:
            terrain_parent = len(gltf.nodes)
            gltf.nodes.append(Node(name="Terrain", children=chunk_nodes))
        else:
            logger.warning("No terrain chunks were loaded!")

    if args.actors_enabled:
        image_indices: Dict[str, int] = {}
        for object in zone.objects:
            # Pass the asset_dir to dme_from_adr_extracted
            dme = dme_from_adr_extracted(asset_dir, object.actor_file)
            if dme is None:
                logger.warning(f"Skipping {object.actor_file}...")
                continue

            if hasattr(dme, 'skipped') and dme.skipped:
                logger.info(f"Skipping {object.actor_file} - marked as skipped (spawner/character)")
                continue

            instances_to_add = [True] * len(object.instances)
            if bounding_box is not None and dme.aabb is not None:
                for i, instance in enumerate(object.instances):
                    # Get original actor coordinates
                    original_translation = numpy.array(astuple(instance.translation)[:3])

                    # Apply the SAME coordinate swapping that terrain uses (but no scaling)
                    # The terrain swaps X/Z coordinates, so we need to do the same
                    # Original: [X, Y, Z] -> Terrain: [Z, Y, X]
                    transformed_translation = numpy.array([
                        original_translation[2],  # Z -> X
                        original_translation[1],  # Y -> Y
                        original_translation[0]  # X -> Z
                    ])

                    transformed_scale = numpy.array(astuple(instance.scale)[:3])

                    # Apply the original rotation (NOT the fixed rotation) for bounding box check
                    original_rotation = Rotation.from_euler("yzx", astuple(instance.rotation)[:3], False)

                    maximum, minimum = None, None
                    for corner in dme.aabb.corners:
                        # Transform using original rotation, not the fixed get_gltf_rotation
                        transformed = transformed_translation + original_rotation.apply(
                            transformed_scale * numpy.array(corner))
                        if maximum is None:
                            maximum = transformed
                        else:
                            maximum = (max(maximum[0], transformed[0]), max(maximum[1], transformed[1]),
                                       max(maximum[2], transformed[2]))
                        if minimum is None:
                            minimum = transformed
                        else:
                            minimum = (min(minimum[0], transformed[0]), min(minimum[1], transformed[1]),
                                       min(minimum[2], transformed[2]))

                    transformed_bbox = AABB(list(zip(minimum, maximum)))

                    # DEBUG: Print first few actor positions
                    if i < 5:
                        logger.info(
                            f"Actor {i}: original={original_translation}, transformed={transformed_translation}")
                        logger.info(f"  bbox_center={[(minimum[j] + maximum[j]) / 2 for j in range(3)]}")
                        logger.info(f"  your bounding box: {bounding_box}")
                        logger.info(f"  Actor bbox: {transformed_bbox}")
                        logger.info(f"  Overlaps: {bounding_box.overlaps(transformed_bbox)}")

                    instances_to_add[i] = bounding_box.overlaps(transformed_bbox)

            if not any(instances_to_add):
                logger.info(f"Skipping {object.actor_file} since no instances are within the bounding box.")
                continue

            node_start = len(gltf.nodes)
            # Pass None for manager since we're using extracted assets
            offset, blob = append_dme_to_gltf_extracted(gltf, dme, asset_dir, mats, textures, image_indices, offset,
                                                        blob,
                                                        object.actor_file, include_skeleton=False)
            node_end = len(gltf.nodes)

            logger.info(f"Adding {sum(instances_to_add)} instances of {object.actor_file}")
            instances = []
            original_nodes_consumed = False
            for i, instance in enumerate(object.instances):
                if not instances_to_add[i]:
                    continue
                if original_nodes_consumed:
                    children = []
                    for node_idx in range(node_start, node_end):
                        children.append(len(gltf.nodes))
                        gltf.nodes.append(Node(mesh=gltf.nodes[node_idx].mesh))
                else:
                    children = list(range(node_start, node_end))
                    original_nodes_consumed = True
                rot = get_gltf_rotation(astuple(instance.rotation)[:3])
                instances.append(len(gltf.nodes))
                gltf.nodes.append(Node(
                    name=Path(object.actor_file).stem,
                    children=children,
                    rotation=rot,
                    translation=astuple(instance.translation)[:3],
                    scale=astuple(instance.scale)[:3]
                ))

            instance_nodes.extend(instances)
        actor_parent = len(gltf.nodes)
        gltf.nodes.append(Node(name="Object Instances", children=instance_nodes))

    if args.lights_enabled:
        if "KHR_lights_punctual" not in gltf.extensionsUsed:
            gltf.extensionsUsed.append("KHR_lights_punctual")
        gltf.extensions["KHR_lights_punctual"] = {
            "lights": []
        }
        light_nodes = []
        light_def_to_index = {}
        logger.info(f"Adding {len(zone.lights)} lights to the scene...")
        for light in zone.lights:
            if bounding_box is not None:
                light_aabb = AABB(list(zip(astuple(light.translation), astuple(light.translation))))
                if not bounding_box.overlaps(light_aabb):
                    logger.debug("Skipping light outside of bounding box...")
                    continue

            light_nodes.append(len(gltf.nodes))
            light_def = {
                "color": [light.color_val.r / 255.0, light.color_val.g / 255.0, light.color_val.b / 255.0],
                "type": "point" if light.type == LightType.Point else "spot",
                "intensity": light.unk_floats[0] * 100  # Intensity value is a guess
            }
            if light.type == LightType.Spot:
                light_def["spot"] = {}  # Add spot properties if needed
            key = str(light_def["color"]) + light_def["type"] + str(light_def["intensity"])
            if key not in light_def_to_index:
                light_def_to_index[key] = len(gltf.extensions["KHR_lights_punctual"]["lights"])
                gltf.extensions["KHR_lights_punctual"]["lights"].append(light_def)

            gltf.nodes.append(Node(
                name=light.name,
                translation=astuple(light.translation)[:3],
                rotation=get_gltf_rotation(astuple(light.rotation)[:3]),
                scale=[1, 1, -1],
                extensions={
                    "KHR_lights_punctual": {
                        "light": light_def_to_index[key]
                    }
                }
            ))
        logger.info(f"Added {len(light_nodes)} instances of {len(light_def_to_index)} unique lights")
        light_parent = len(gltf.nodes)
        gltf.nodes.append(Node(name="Lights", children=light_nodes))

    gltf.buffers.append(Buffer(
        byteLength=offset
    ))

    scene_nodes = []
    if terrain_parent is not None:
        scene_nodes.append(terrain_parent)
    if actor_parent is not None:
        scene_nodes.append(actor_parent)
    if light_parent is not None:
        scene_nodes.append(light_parent)
    gltf.scene = 0
    gltf.scenes.append(Scene(nodes=scene_nodes))

    logger.info("Saving GLTF file...")
    if args.format == "glb":
        gltf.set_binary_blob(blob)
        gltf.save_binary(args.output_file)
    elif args.format == "gltf":
        blobpath = Path(args.output_file).with_suffix(".bin")
        with open(blobpath, "wb") as f:
            f.write(blob)
        gltf.buffers[0].uri = blobpath.name
        gltf.save_json(args.output_file)

    if not args.skip_textures:
        logger.info("Saving Textures...")
        save_textures(args.output_file, textures)
        logger.info(f"Saved {len(textures)} textures")

    from dme_loader.dme_loader import finalize_material_logging
    log_file = finalize_material_logging()
    if log_file:
        print(f"\nMaterial choice summary saved to: {log_file}")


if __name__ == "__main__":
    main()