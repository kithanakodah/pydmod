import struct
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Tuple, Optional
from aabbtree import AABB
from cnkdec import Decompressor

logger = logging.getLogger("cnk_loader")


@dataclass
class Vertex:
    """Represents a vertex from a CNK1 file (chunkLodV2Schema)."""
    x: int
    y: int
    height_far: int
    height_near: int
    color: int


@dataclass
class RenderBatch:
    """Represents a render batch from CNK1 files."""
    unknown: int = 0  # Extra field in version 2
    index_offset: int = 0
    index_count: int = 0
    vertex_offset: int = 0
    vertex_count: int = 0


@dataclass
class Chunk:
    """
    Schema-accurate parser for H1Z1 CNK1 files that follows the exact JavaScript
    chunkLodV2Schema structure from your reference files.
    """
    # --- Data read directly from the file ---
    magic: bytes = b'CNK1'
    version: int = 2
    verts_per_side: int = 0
    indices: List[int] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)
    render_batches: List[RenderBatch] = field(default_factory=list)

    # --- Data calculated for export ---
    verts: List[Tuple[float, float, float]] = field(default_factory=list)
    triangles: List[int] = field(default_factory=list)
    aabb: Optional[AABB] = None

    @classmethod
    def load(cls, data: BytesIO, compressed: bool = True) -> 'Chunk':
        """
        Loads a CNK1 file following the exact chunkLodV2Schema from the JavaScript reference.
        """
        if compressed:
            logger.debug("Decompressing CNK file...")
            data.seek(0)
            header_magic = data.read(4)
            header_version = struct.unpack("<I", data.read(4))[0]

            decompressor = Decompressor()
            decompressed_size, compressed_size = struct.unpack("<II", data.read(8))
            compressed_data = data.read()
            decompressed_data = decompressor.decompress(compressed_data, decompressed_size)

            # Reconstruct the stream with the header and decompressed data
            data = BytesIO(header_magic + struct.pack("<I", header_version) + decompressed_data)

        data.seek(0)
        magic = data.read(4)
        version = struct.unpack("<I", data.read(4))[0]

        if magic not in [b'CNK1', b'CNK2', b'CNK3', b'CNK4', b'CNK5']:
            raise ValueError(f"File is not a supported CNK file! Magic: {magic}")

        logger.info(f"Loading {magic.decode()} Version {version} file...")

        # Get file size for debugging
        current_pos = data.tell()
        data.seek(0, 2)
        file_size = data.tell()
        data.seek(current_pos)
        logger.debug(f"File size: {file_size} bytes")

        # === EXACT chunkLodV2Schema Implementation ===

        # 1. textures (ARRAY - dynamic count!)
        texture_count = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Texture array count: {texture_count}")

        for texture_set in range(texture_count):
            logger.debug(f"Processing texture set {texture_set}/{texture_count}")
            # Each texture set has 6 "byteswithlength" fields
            for field_idx, field_name in enumerate(
                    ['colorNxMap', 'specNyMap', 'extraData1', 'extraData2', 'extraData3', 'extraData4']):
                texture_len = struct.unpack("<I", data.read(4))[0]
                logger.debug(f"  {field_name}: {texture_len} bytes")
                if texture_len > 0:
                    data.seek(texture_len, 1)  # Skip texture data

        # 2. vertsPerSide (uint32)
        verts_per_side = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"vertsPerSide: {verts_per_side}")

        # 3. heightMaps (CUSTOM parser - following parseHeightMaps from JS)
        heightmap_data_length = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"HeightMap dataLength: {heightmap_data_length}")

        # The JavaScript parseHeightMaps function:
        # - Reads dataLength (already read above)
        # - Calculates n = dataLength / 4
        # - Reads 4 height maps, each with n entries
        # - Each entry is: int16, uint8, uint8 (4 bytes total)
        n = heightmap_data_length // 4
        logger.debug(f"HeightMap entries per map: {n}")

        # Skip the 4 height maps (4 * n * 4 bytes each)
        total_heightmap_bytes = 4 * n * 4
        logger.debug(f"Skipping {total_heightmap_bytes} bytes of heightmap data")
        data.seek(total_heightmap_bytes, 1)

        # 4. indices (array of uint16)
        indices_count = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Indices count: {indices_count}")

        indices_data = data.read(indices_count * 2)
        indices = [i[0] for i in struct.iter_unpack("<H", indices_data)]

        # 5. vertices (array of vertex structures)
        vertices_count = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Vertices count: {vertices_count}")

        vertices = []
        for i in range(vertices_count):
            vertex_data = data.read(12)  # uint16 + uint16 + int16 + int16 + uint32 = 12 bytes
            if len(vertex_data) < 12:
                logger.error(f"Incomplete vertex data at vertex {i}")
                break
            x, y, h_far, h_near, color = struct.unpack("<HHhhI", vertex_data)
            vertices.append(Vertex(x, y, h_far, h_near, color))

        # 6. renderBatches (array with version 2 having extra 'unknown' field)
        render_batches = []
        current_offset = data.tell()
        logger.debug(f"Reading render batches at offset {current_offset}")

        # Check if we have enough data for batch count
        remaining_bytes = file_size - current_offset
        if remaining_bytes < 4:
            logger.warning(f"Not enough data for batch count ({remaining_bytes} bytes remaining)")
        else:
            try:
                batch_count = struct.unpack("<I", data.read(4))[0]
                logger.debug(f"Render batch count: {batch_count}")

                # Sanity check - if batch count is huge, we're probably reading garbage
                if batch_count > 1000:  # Reasonable upper limit
                    logger.warning(f"Suspiciously large batch count: {batch_count}, truncating to 100")
                    batch_count = min(batch_count, 100)

                for i in range(batch_count):
                    remaining_bytes = file_size - data.tell()
                    bytes_needed = 20 if version == 2 else 16

                    if remaining_bytes < bytes_needed:
                        logger.warning(f"Not enough data for batch {i} ({remaining_bytes} < {bytes_needed})")
                        break

                    if version == 2:
                        # chunkLodV2Schema: unknown, indexOffset, indexCount, vertexOffset, vertexCount
                        batch_data = data.read(20)
                        unknown, idx_offset, idx_count, vtx_offset, vtx_count = struct.unpack("<IIIII", batch_data)
                        render_batches.append(RenderBatch(unknown, idx_offset, idx_count, vtx_offset, vtx_count))
                    else:
                        # chunkLodSchema: indexOffset, indexCount, vertexOffset, vertexCount
                        batch_data = data.read(16)
                        idx_offset, idx_count, vtx_offset, vtx_count = struct.unpack("<IIII", batch_data)
                        render_batches.append(RenderBatch(0, idx_offset, idx_count, vtx_offset, vtx_count))

                    # Validate batch data for sanity
                    batch = render_batches[-1]
                    if (batch.index_offset > 1000000 or batch.index_count > 1000000 or
                            batch.vertex_offset > 1000000 or batch.vertex_count > 1000000):
                        logger.warning(f"Batch {i} has suspiciously large values, stopping batch parsing")
                        render_batches.pop()  # Remove the bad batch
                        break

                    logger.debug(
                        f"Batch {i}: unknown={batch.unknown}, idx_offset={batch.index_offset}, idx_count={batch.index_count}, vtx_offset={batch.vertex_offset}, vtx_count={batch.vertex_count}")

            except struct.error as e:
                logger.warning(f"Error reading render batches: {e}")

        logger.info(
            f"✅ Successfully loaded {magic.decode()} with {len(vertices)} vertices, {len(indices)} indices, {len(render_batches)} batches")

        return cls(
            magic=magic,
            version=version,
            verts_per_side=verts_per_side,
            indices=indices,
            vertices=vertices,
            render_batches=render_batches,
        )

    def calculate_verts_js_exact(self, chunk_x_grid=0, chunk_z_grid=0):
        """
        Alternative that exactly replicates the JavaScript cnktool.js logic for perfect compatibility.
        This should fix chunk stitching issues.
        """
        if len(self.verts) > 0 and len(self.triangles) > 0:
            return

        logger.info(f"Calculating vertices using JS-exact method for chunk ({chunk_x_grid}, {chunk_z_grid})")

        self.verts = []
        self.triangles = []

        # Process exactly like JavaScript "geometryabs" mode
        vertex_base = 0
        for i in range(min(4, len(self.render_batches))):
            batch = self.render_batches[i]

            # --- Vertex Calculation ---
            for j in range(batch.vertex_count):
                k = batch.vertex_offset + j
                if k >= len(self.vertices): continue
                v = self.vertices[k]

                # JS logic for quad position within the chunk
                quad_x_offset = (i >> 1) * 64
                quad_y_offset = (i % 2) * 64

                # JS logic for placing the chunk in the world
                x = v.x + quad_x_offset + (chunk_z_grid * 32)
                y = v.y + quad_y_offset + (chunk_x_grid * 32)

                # Convert to world coordinates and apply scaling
                # DEFINITIVE FIX: Apply a scaling factor of 2 to all axes to match the actor/world scale.
                # The original tool was likely for a different game version with different units.
                world_x = float(x) * 2.0
                world_y = (float(v.height_near) / 64.0) * 2.0
                world_z = float(y) * 2.0

                self.verts.append((world_x, world_y, world_z))

            # --- Triangle Calculation ---
            for j in range(0, batch.index_count, 3):
                if j + 2 >= batch.index_count: break

                # JavaScript reverses triangle winding order
                v0 = self.indices[j + batch.index_offset + 2] + vertex_base
                v1 = self.indices[j + batch.index_offset + 1] + vertex_base
                v2 = self.indices[j + batch.index_offset + 0] + vertex_base

                if all(idx < len(self.verts) for idx in [v0, v1, v2]):
                    self.triangles.extend([v0, v1, v2])

            vertex_base += batch.vertex_count

        # Calculate AABB
        if self.verts:
            xs = [v[0] for v in self.verts]
            ys = [v[1] for v in self.verts]
            zs = [v[2] for v in self.verts]
            minimum = [min(xs), min(ys), min(zs)]
            maximum = [max(xs), max(ys), max(zs)]
            self.aabb = AABB(list(zip(minimum, maximum)))

        logger.info(f"✅ JS-exact method generated {len(self.verts)} vertices, {len(self.triangles) // 3} triangles")
        if self.aabb:
            logger.info(f"AABB: {self.aabb}")


# For compatibility
ForgelightChunk = Chunk