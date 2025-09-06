import multiprocessing
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

from argparse import ArgumentParser
from DbgPack import AssetManager
from io import SEEK_END, BytesIO, FileIO, StringIO
from typing import Optional, Tuple

from dme_loader import DME, DMAT
from dme_converter import get_manager, to_glb, to_gltf

logger = logging.getLogger("ADR Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))


def load_adr(file: FileIO) -> Optional[ET.Element]:
    tree = ET.parse(file)
    root = tree.getroot()
    if root.tag != "ActorRuntime":
        logger.error("File's root XML tag was not ActorRuntime!")
        return None
    return root


def get_base_model(root: ET.Element) -> Optional[Tuple[str, str]]:
    base = root.find("Base")
    if base is None:
        logger.error("No base model present in Actor Runtime file")
        return None
    return base.get("fileName"), base.get("paletteName")


def get_animation_network(root: ET.Element) -> Optional[str]:
    anim_network = root.find("AnimationNetwork")
    if anim_network is None:
        logger.warning("No AnimationNetwork present in Actor Runtime file")
        return None
    return anim_network.get("fileName")


def dme_from_adr(manager: AssetManager, adr_file: str, asset_dir: Path = None) -> Optional[DME]:
    """Load DME from ADR file, with optional extracted assets directory"""
    logger.info(f"Loading ADR file {adr_file}...")

    # Try extracted assets first if asset_dir is provided
    if asset_dir:
        extracted_file = asset_dir / adr_file
        try:
            if extracted_file.exists():
                logger.info(f"Loading from extracted assets: {extracted_file}")
                file = open(extracted_file)
            else:
                file = open(adr_file)
        except FileNotFoundError:
            logger.error(f"File not found: {adr_file}")
            return None
    else:
        # Original behavior for pack files
        try:
            file = open(adr_file)
        except FileNotFoundError:
            logger.warning(f"File not found: {adr_file}. Loading from game assets...")
            if not manager.loaded.is_set():
                logger.info("Waiting for assets to load...")
            manager.loaded.wait()
            adr_asset = manager.get_raw(adr_file)
            if adr_asset is None:
                logger.error(f"{adr_file} not found in either game assets or filesystem")
                return None
            file = StringIO(adr_asset.get_data().decode("utf-8"))

    root = load_adr(file)
    file.close()
    if root is None:
        return None
    dme_props = get_base_model(root)
    if dme_props is None:
        return None
    dme_name, dma_palette = dme_props

    # Load DME file
    if asset_dir:
        # Try loading DME from extracted assets first
        extracted_dme_file = asset_dir / dme_name
        if extracted_dme_file.exists():
            logger.info(f"Loading DME from extracted assets: {extracted_dme_file}")
            with open(extracted_dme_file, 'rb') as dme_file:
                dme = DME.load(dme_file, adr_file=adr_file, dme_file=dme_name)
        else:
            logger.error(f"DME file not found in extracted assets: {extracted_dme_file}")
            return None
    else:
        # Fall back to pack files
        if not manager.loaded.is_set():
            logger.info("Waiting for assets to load...")
        manager.loaded.wait()
        dme_asset = manager.get_raw(dme_name)
        if dme_asset is None:
            logger.error(f"Could not find {dme_name} in loaded assets")
            return None
        dme_file = BytesIO(dme_asset.get_data())
        dme = DME.load(dme_file, adr_file=adr_file, dme_file=dme_name)
        dme_file.close()

    dme.name = dme_name

    # Handle palette/material files
    dmat: DMAT = None
    if Path(dma_palette).stem.lower() != Path(dme_name).stem.lower():
        logger.info(f"Loading palette with different name: {Path(dma_palette).stem} vs {Path(dme_name).stem}")

        if asset_dir:
            # Try loading DMA from extracted assets first
            extracted_dma_file = asset_dir / dma_palette
            if extracted_dma_file.exists():
                logger.info(f"Loading DMA from extracted assets: {extracted_dma_file}")
                with open(extracted_dma_file, 'rb') as dma_file:
                    dma_data = dma_file.read()
                    dma_file = BytesIO(dma_data)
                    logger.info(f"Loaded palette asset with length {len(dma_data)}, creating DMAT...")
                    dmat = DMAT.load(dma_file, len(dma_data))
        else:
            # Fall back to pack files
            dma_asset = manager.get_raw(dma_palette)
            if dma_asset:
                dma_file = BytesIO(dma_asset.get_data())
                logger.info(f"Loaded palette asset with length {len(dma_asset.get_data())}, creating DMAT...")
                dmat = DMAT.load(dma_file, len(dma_asset.get_data()))

    if dmat is not None:
        dme.dmat = dmat
    return dme


def main():
    parser = ArgumentParser(description="Actor Runtime (.adr) to gltf/glb converter")
    parser.add_argument("input_file", type=str, help="Path of the input ADR file")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"],
                        help="The output format to use, required for conversion")
    parser.add_argument("--asset-dir", type=str,
                        help="Path to the extracted H1Z1 assets directory (if not using pack files)")
    parser.add_argument("--live", "-l", action="store_true", help="Load assets from live server rather than test")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count",
                        default=0)
    parser.add_argument("--no-skeleton", "-n", action="store_true", help="Exclude skeleton from generated mesh",
                        default=False)
    args = parser.parse_args()

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    asset_dir = Path(args.asset_dir) if args.asset_dir else None

    if asset_dir:
        # Using extracted assets - no need for pack file manager
        dme = dme_from_adr(None, args.input_file, asset_dir)
        if args.format == "gltf":
            to_gltf_extracted(dme, args.output_file, asset_dir, dme.name, not args.no_skeleton)
        elif args.format == "glb":
            to_glb_extracted(dme, args.output_file, asset_dir, dme.name, not args.no_skeleton)
    else:
        # Using pack files - original behavior
        with multiprocessing.Pool(8) as pool:
            manager = get_manager(pool, args.live)
            dme = dme_from_adr(manager, args.input_file)
            if args.format == "gltf":
                to_gltf(dme, args.output_file, manager, dme.name, not args.no_skeleton)
            elif args.format == "glb":
                to_glb(dme, args.output_file, manager, dme.name, not args.no_skeleton)


def to_gltf_extracted(dme: DME, output: str, asset_dir: Path, dme_name: str, include_skeleton: bool = True):
    """Convert DME to GLTF using extracted assets"""
    from dme_converter import dme_to_gltf_extracted, save_textures

    gltf, blob, textures = dme_to_gltf_extracted(dme, asset_dir, dme_name, str(Path(output).stem), include_skeleton)
    blobpath = Path(output).with_suffix(".bin")
    with open(blobpath, "wb") as blob_out:
        blob_out.write(blob)
    gltf.buffers[0].uri = blobpath.name
    gltf.save_json(output)
    save_textures(output, textures)


def to_glb_extracted(dme: DME, output: str, asset_dir: Path, dme_name: str, include_skeleton: bool = True):
    """Convert DME to GLB using extracted assets"""
    from dme_converter import dme_to_gltf_extracted, save_textures

    gltf, blob, textures = dme_to_gltf_extracted(dme, asset_dir, dme_name, str(Path(output).stem), include_skeleton)
    gltf.set_binary_blob(blob)
    gltf.save_binary(output)
    save_textures(output, textures)


if __name__ == "__main__":
    main()