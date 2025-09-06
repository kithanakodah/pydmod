const fs = require('fs');

function readCNK0(filePath) {
    console.log("Reading:", filePath);
    
    const data = fs.readFileSync(filePath);
    console.log("File size:", data.length, "bytes");
    
    // Read header
    const magic = data.toString('ascii', 0, 4);
    const version = data.readUInt32LE(4);
    const tileCount = data.readUInt32LE(8);
    
    console.log("Magic:", magic);
    console.log("Version:", version);
    console.log("Tiles:", tileCount);
    
    if (magic !== 'CNK0') {
        console.log("ERROR: Not a CNK0 file");
        return;
    }
    
    // Search for geometry data
    console.log("\nSearching for geometry data...");
    
    let found = false;
    
    // Try different offsets
    const testOffsets = [
        2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000
    ];
    
    for (let baseOffset of testOffsets) {
        if (baseOffset >= data.length - 100) continue;
        
        try {
            const val1 = data.readUInt32LE(baseOffset);
            const val2 = data.readUInt32LE(baseOffset + 4);
            
            // Skip unknown array
            const nextOffset = baseOffset + 8 + (val2 * 4);
            
            if (nextOffset + 4 >= data.length) continue;
            
            const indicesCount = data.readUInt32LE(nextOffset);
            
            if (indicesCount < 1000 || indicesCount > 100000) continue;
            
            const verticesOffset = nextOffset + 4 + (indicesCount * 2);
            
            if (verticesOffset + 4 >= data.length) continue;
            
            const verticesCount = data.readUInt32LE(verticesOffset);
            
            if (verticesCount < 500 || verticesCount > 50000) continue;
            
            console.log("\n*** FOUND GEOMETRY DATA ***");
            console.log("Base offset:", baseOffset);
            console.log("unk1:", val1);
            console.log("unk array count:", val2);
            console.log("Indices count:", indicesCount);
            console.log("Vertices count:", verticesCount);
            
            // Read some sample data
            console.log("\nSample indices:");
            const indicesStart = nextOffset + 4;
            const sampleIndices = [];
            
            for (let i = 0; i < Math.min(10, indicesCount); i++) {
                const idx = data.readUInt16LE(indicesStart + (i * 2));
                sampleIndices.push(idx);
            }
            console.log(sampleIndices.join(', '));
            
            console.log("\nSample vertices:");
            const verticesStart = verticesOffset + 4;
            
            for (let i = 0; i < Math.min(3, verticesCount); i++) {
                const vOffset = verticesStart + (i * 16);
                const x = data.readInt16LE(vOffset);
                const y = data.readInt16LE(vOffset + 2);
                const hFar = data.readInt16LE(vOffset + 4);
                const hNear = data.readInt16LE(vOffset + 6);
                const c1 = data.readUInt32LE(vOffset + 8);
                const c2 = data.readUInt32LE(vOffset + 12);
                
                console.log(`Vertex ${i}: x=${x}, y=${y}, hNear=${hNear}, hFar=${hFar}`);
            }
            
            // Try to find render batches
            const batchOffset = verticesStart + (verticesCount * 16);
            if (batchOffset + 4 < data.length) {
                const batchCount = data.readUInt32LE(batchOffset);
                console.log("\nRender batch count:", batchCount);
                
                if (batchCount > 0 && batchCount <= 10) {
                    for (let i = 0; i < batchCount; i++) {
                        const bOffset = batchOffset + 4 + (i * 16);
                        if (bOffset + 16 <= data.length) {
                            const indexOffset = data.readUInt32LE(bOffset);
                            const indexCount = data.readUInt32LE(bOffset + 4);
                            const vertexOffset = data.readUInt32LE(bOffset + 8);
                            const vertexCount = data.readUInt32LE(bOffset + 12);
                            
                            console.log(`Batch ${i}: idxOff=${indexOffset}, idxCnt=${indexCount}, vtxOff=${vertexOffset}, vtxCnt=${vertexCount}`);
                        }
                    }
                }
            }
            
            // Save the successful parsing info
            const result = {
                magic: magic,
                version: version,
                tileCount: tileCount,
                geometryOffset: baseOffset,
                indicesCount: indicesCount,
                verticesCount: verticesCount,
                sampleIndices: sampleIndices
            };
            
            fs.writeFileSync('./FOUND_GEOMETRY.json', JSON.stringify(result, null, 2));
            console.log("\n✅ Geometry data structure saved to FOUND_GEOMETRY.json");
            
            found = true;
            break;
            
        } catch (e) {
            // Continue to next offset
        }
    }
    
    if (!found) {
        console.log("❌ Could not find geometry data");
    }
}

// Run it
if (process.argv.length < 3) {
    console.log("Usage: node simple_cnk_reader.js <cnk0_file>");
    process.exit(1);
}

readCNK0(process.argv[2]);