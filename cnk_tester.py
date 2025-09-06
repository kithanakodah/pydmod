import logging
from pathlib import Path
from time import sleep

from cnk_loader import ForgelightChunk
from io import BytesIO

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.DEBUG, handlers=[handler])


def load_chunk_from_extracted(base_path: Path, chunk_name: str) -> bytes:
    """Load chunk data from extracted files"""
    chunk_path = base_path / chunk_name
    if chunk_path.exists():
        with open(chunk_path, 'rb') as f:
            data = f.read()
        print(f"Loaded {chunk_name}: {len(data)} bytes")
        return data
    else:
        print(f"Chunk file not found: {chunk_path}")
    return None


def main():
    chunk_name = "Z1_-4_-4.cnk0"
    extracted_path = Path("M:/H1Z1_assets")
    
    print(f"Testing {chunk_name}...")
    
    # Load the chunk data
    chunk_data_bytes = load_chunk_from_extracted(extracted_path, chunk_name)
    
    if chunk_data_bytes is None:
        print("❌ Could not find chunk file")
        return
    
    print(f"✅ Found chunk: {len(chunk_data_bytes)} bytes")
    
    # Check magic bytes
    if len(chunk_data_bytes) >= 4:
        magic = chunk_data_bytes[:4]
        print(f"Magic bytes: {magic} (hex: {magic.hex()})")
    
    # Check first 32 bytes
    print(f"First 32 bytes: {chunk_data_bytes[:32].hex()}")
    
    # Try to load it
    try:
        print("Attempting to load chunk...")
        chunk = ForgelightChunk.load(BytesIO(chunk_data_bytes))
        print(f"✅ Successfully loaded chunk!")
        print(f"Chunk type: {type(chunk)}")
        
        # Try to calculate vertices
        print("Calculating vertices...")
        chunk.calculate_verts()
        print(f"✅ Successfully calculated vertices")
        
        if hasattr(chunk, 'verts'):
            print(f"Vertex count: {len(chunk.verts)}")
            
        # ADD THIS DEBUGGING CODE:
        if hasattr(chunk, 'verts') and len(chunk.verts) > 0:
            print(f"✅ Got {len(chunk.verts)} vertices")
            print(f"Render batches: {len(chunk.render_batches)}")
            print(f"Indices: {len(chunk.indices)}")
            print(f"Triangles: {len(chunk.triangles)}")
            
            # Look at first few vertices
            print("First 5 vertices:")
            for i in range(min(5, len(chunk.verts))):
                print(f"  Vertex {i}: {chunk.verts[i]}")
                
            # Check render batch info
            print("First 3 render batches:")
            for i in range(min(3, len(chunk.render_batches))):
                batch = chunk.render_batches[i]
                print(f"  Batch {i}: offset={batch.vertex_offset}, count={batch.vertex_count}, indices_offset={batch.index_offset}, indices_count={batch.index_count}")
        
    except Exception as e:
        print(f"❌ Failed to load chunk: {e}")
        import traceback
        traceback.print_exc()
    
    sleep(2)


if __name__ == "__main__":
    main()