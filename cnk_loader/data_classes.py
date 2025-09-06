import struct
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple
from aabbtree import AABB

logger = logging.getLogger("simple_cnk")


@dataclass
class SimpleVertex:
    x: int
    y: int
    height_far: int
    height_near: int
    color1: int
    color2: int


@dataclass
class SimpleCNK0:
    """
    Simple CNK0 parser that just extracts geometry data without complex parsing
    Based on the successful JavaScript analysis that found data at known offsets
    """

    vertices: List[SimpleVertex]
    indices: List[int]
    verts: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]
    triangles: List[List[int]]
    aabb: AABB

    @classmethod
    def load(cls, data: BytesIO, compressed: bool = False) -> 'SimpleCNK0':
        """Load CNK0 by finding geometry data and extracting it directly"""

        if compressed:
            raise ValueError("This loader only supports decompressed CNK0 files")

        logger.info("Loading CNK0 using simple approach - finding geometry data...")

        # Read all data
        data.seek(0)
        file_data = data.read()

        logger.info(f"Analyzing {len(file_data):,} bytes...")

        # Look for the geometry pattern we found in JavaScript analysis
        # Pattern: reasonable array count, then indices count, then vertices count

        vertices = []
        indices = []
        found = False

        # Search for the pattern - be more aggressive and permissive
        for offset in range(8, len(file_data) - 1000, 1):  # Search every byte, not every 4 bytes
            try:
                # Read potential structure: [unk1] [array_count] [array_data...] [indices_count] [indices...] [vertices_count] [vertices...]
                unk1 = struct.unpack("<I", file_data[offset:offset + 4])[0]
                array_count = struct.unpack("<I", file_data[offset + 4:offset + 8])[0]

                # Look for reasonable array count - be more permissive
                if 0 < array_count < 100000:  # Increased upper limit
                    # Calculate where indices should start
                    indices_start = offset + 8 + (array_count * 4)

                    if indices_start + 4 <= len(file_data):
                        indices_count = struct.unpack("<I", file_data[indices_start:indices_start + 4])[0]

                        # Look for reasonable indices count - be more permissive
                        if 100 <= indices_count <= 300000:  # Lowered minimum, increased maximum
                            # Calculate where vertices should start
                            vertices_start = indices_start + 4 + (indices_count * 2)

                            if vertices_start + 4 <= len(file_data):
                                vertices_count = struct.unpack("<I", file_data[vertices_start:vertices_start + 4])[0]

                                # Look for reasonable vertex count - be more permissive
                                if 50 <= vertices_count <= 200000:  # Lowered minimum, increased maximum
                                    logger.info(f"🎯 Found geometry at offset {offset}:")
                                    logger.info(f"  unk1: {unk1}, array_count: {array_count}")
                                    logger.info(f"  indices_count: {indices_count}, vertices_count: {vertices_count}")

                                    # Read the indices
                                    indices = []
                                    indices_pos = indices_start + 4
                                    for i in range(indices_count):
                                        if indices_pos + 2 <= len(file_data):
                                            idx = struct.unpack("<H", file_data[indices_pos:indices_pos + 2])[0]
                                            indices.append(idx)
                                            indices_pos += 2
                                        else:
                                            break

                                    # Read the vertices
                                    vertices = []
                                    vertices_pos = vertices_start + 4
                                    for i in range(vertices_count):
                                        if vertices_pos + 16 <= len(file_data):
                                            try:
                                                x, y, h_far, h_near, c1, c2 = struct.unpack("<hhhhII",
                                                                                            file_data[
                                                                                            vertices_pos:vertices_pos + 16])
                                                vertices.append(SimpleVertex(x, y, h_far, h_near, c1, c2))
                                                vertices_pos += 16
                                            except struct.error:
                                                break
                                        else:
                                            break

                                    logger.info(
                                        f"✅ Successfully extracted {len(vertices)} vertices, {len(indices)} indices")
                                    found = True
                                    break
            except (struct.error, IndexError):
                continue

        if not found:
            logger.warning("Could not find geometry data - creating empty chunk")
            vertices = []
            indices = []

        return cls(
            vertices=vertices,
            indices=indices,
            verts=[],
            uvs=[],
            triangles=[],
            aabb=AABB()
        )

    def calculate_verts(self, chunk_x_grid=0, chunk_z_grid=0):
        """Calculate world coordinates using simple approach"""
        if self.verts or not self.vertices:
            return

        logger.info(f"Calculating vertices for chunk at grid ({chunk_x_grid}, {chunk_z_grid})")
        logger.info(f"Processing {len(self.vertices)} vertices and {len(self.indices)} indices")

        # World coordinate calculation
        world_origin_x = chunk_z_grid * 32.0
        world_origin_z = chunk_x_grid * 32.0

        self.verts = []
        self.uvs = []
        self.triangles = []

        minimum, maximum = [float('inf')] * 3, [float('-inf')] * 3

        # Process all vertices directly
        for i, v in enumerate(self.vertices):
            # Apply world coordinates directly
            world_x = world_origin_x + v.x
            world_y = v.height_near / 64.0
            world_z = world_origin_z + v.y

            self.verts.append((world_x, world_y, world_z))

            # Generate UV coordinates
            uv_u = v.y / 128.0
            uv_v = 1.0 - v.x / 128.0
            self.uvs.append((uv_u, uv_v))

            # Update bounding box
            minimum[0] = min(minimum[0], world_x)
            minimum[1] = min(minimum[1], world_y)
            minimum[2] = min(minimum[2], world_z)
            maximum[0] = max(maximum[0], world_x)
            maximum[1] = max(maximum[1], world_y)
            maximum[2] = max(maximum[2], world_z)

        # Create triangles from indices
        triangles = []
        triangle_count = 0

        for i in range(0, len(self.indices), 3):
            if i + 2 < len(self.indices):
                v0_idx = self.indices[i]
                v1_idx = self.indices[i + 1]
                v2_idx = self.indices[i + 2]

                # Check bounds
                if (v0_idx < len(self.vertices) and
                        v1_idx < len(self.vertices) and
                        v2_idx < len(self.vertices)):
                    # Add triangle indices (reversed for correct winding)
                    triangles.extend([v2_idx, v1_idx, v0_idx])
                    triangle_count += 1

        if triangles:
            self.triangles.append(triangles)

        if minimum[0] != float('inf'):
            self.aabb = AABB(list(zip(minimum, maximum)))

        logger.info(f"✅ Generated {len(self.verts)} vertices, {triangle_count} triangles")
        if self.aabb:
            logger.info(f"AABB: {self.aabb}")


# For compatibility
CNK0 = SimpleCNK0
ForgelightChunk = SimpleCNK0


# Dummy classes for backward compatibility
class Vertex:
    def __init__(self, x=0, y=0, height_far=0, height_near=0, color1=0, color2=0):
        self.x = x
        self.y = y
        self.height_far = height_far
        self.height_near = height_near
        self.color1 = color1
        self.color2 = color2


class RenderBatch:
    def __init__(self, index_offset=0, index_count=0, vertex_offset=0, vertex_count=0):
        self.index_offset = index_offset
        self.index_count = index_count
        self.vertex_offset = vertex_offset
        self.vertex_count = vertex_count


class Tile:
    def __init__(self, x=0, y=0, **kwargs):
        self.x = x
        self.y = y


class Eco:
    def __init__(self, **kwargs):
        pass


class Flora:
    def __init__(self, **kwargs):
        pass


class Layer:
    def __init__(self, **kwargs):
        pass


class Tiles:
    def __init__(self, tiles=None):
        self.tiles = tiles or []

    def __iter__(self): return iter(self.tiles)

    def __len__(self): return len(self.tiles)

    def __getitem__(self, index): return self.tiles[index]


class Header:
    def __init__(self, magic=b'CNK0', version=0):
        self.magic = magic
        self.version = version