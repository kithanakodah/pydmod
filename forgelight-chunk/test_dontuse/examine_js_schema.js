const fs = require('fs');

console.log("=== EXAMINING JAVASCRIPT CNK SCHEMA ===\n");

// Read the JavaScript files to understand the schema
const files = ['cnk.js', 'cnktool.js', 'dataschema.js'];

for (const file of files) {
    if (fs.existsSync(file)) {
        console.log(`\n=== ${file.toUpperCase()} ===`);
        const content = fs.readFileSync(file, 'utf8');
        
        if (file === 'cnk.js') {
            console.log("Looking for schema definitions...");
            
            // Extract schema definitions
            const lines = content.split('\n');
            let inSchema = false;
            let schemaLines = [];
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                
                // Look for schema definitions
                if (line.includes('Schema') || line.includes('schema') || line.includes('fields')) {
                    console.log(`Line ${i + 1}: ${line.trim()}`);
                    inSchema = true;
                    schemaLines = [line];
                } else if (inSchema) {
                    schemaLines.push(line);
                    console.log(`Line ${i + 1}: ${line.trim()}`);
                    
                    // Stop when we hit a closing bracket or new function
                    if (line.includes('};') || line.includes('function') || schemaLines.length > 50) {
                        inSchema = false;
                        console.log("--- End of schema section ---\n");
                    }
                }
                
                // Also look for readChunk function
                if (line.includes('readChunk') || line.includes('function read')) {
                    console.log(`PARSE FUNCTION - Line ${i + 1}: ${line.trim()}`);
                }
            }
        }
        
        if (file === 'cnktool.js') {
            console.log("Looking for geometry processing...");
            
            const lines = content.split('\n');
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                
                // Look for geometry processing
                if (line.includes('vertices') || line.includes('indices') || line.includes('renderBatch') || 
                    line.includes('geometryabs') || line.includes('chunk.')) {
                    console.log(`Line ${i + 1}: ${line.trim()}`);
                }
            }
        }
    } else {
        console.log(`❌ ${file} not found`);
    }
}

console.log("\n=== CREATING MINIMAL CNK PARSER BASED ON INSPECTION ===");

// Let's try to create a parser based on what we know about the format
function parseMinimalCNK0(filePath) {
    console.log(`\nParsing: ${filePath}`);
    
    const data = fs.readFileSync(filePath);
    let offset = 0;
    
    // Read header
    const magic = data.toString('ascii', 0, 4);
    const version = data.readUInt32LE(4);
    offset = 8;
    
    console.log(`Header: ${magic} v${version}`);
    
    if (magic !== 'CNK0') {
        console.log("Not a CNK0 file");
        return;
    }
    
    // Based on the JavaScript schema pattern, CNK0 should have:
    // 1. Tiles section
    // 2. Unknown data
    // 3. Indices 
    // 4. Vertices
    // 5. Render batches
    
    console.log("Trying to parse as CNK0 format...");
    
    // Read tiles count
    if (offset + 4 <= data.length) {
        const tileCount = data.readUInt32LE(offset);
        offset += 4;
        console.log(`Tiles: ${tileCount}`);
        
        if (tileCount > 0 && tileCount <= 100) {
            // Skip tiles - this is the tricky part that's causing issues
            console.log("Skipping tile section (complex variable-length data)...");
            
            // Instead of parsing tiles, let's search for the pattern that comes after tiles
            // We're looking for: [int] [small_count] [data...] [medium_count] [indices...] [medium_count] [vertices...]
            
            const searchStart = offset;
            let found = false;
            
            // More comprehensive search
            for (let searchOffset = searchStart; searchOffset < data.length - 1000; searchOffset += 1) {
                try {
                    // Look for the pattern: unk1, small_array_count, then indices_count, then vertices_count
                    const unk1 = data.readUInt32LE(searchOffset);
                    const arrayCount = data.readUInt32LE(searchOffset + 4);
                    
                    if (arrayCount > 0 && arrayCount < 10000) {
                        const afterArrayOffset = searchOffset + 8 + (arrayCount * 4);
                        
                        if (afterArrayOffset + 8 <= data.length) {
                            const potentialIndicesCount = data.readUInt32LE(afterArrayOffset);
                            
                            if (potentialIndicesCount >= 1000 && potentialIndicesCount <= 200000) {
                                const afterIndicesOffset = afterArrayOffset + 4 + (potentialIndicesCount * 2);
                                
                                if (afterIndicesOffset + 4 <= data.length) {
                                    const potentialVerticesCount = data.readUInt32LE(afterIndicesOffset);
                                    
                                    if (potentialVerticesCount >= 500 && potentialVerticesCount <= 100000) {
                                        console.log(`\n🎯 FOUND POTENTIAL GEOMETRY AT OFFSET ${searchOffset}:`);
                                        console.log(`  unk1: ${unk1}`);
                                        console.log(`  array_count: ${arrayCount}`);
                                        console.log(`  indices_count: ${potentialIndicesCount}`);
                                        console.log(`  vertices_count: ${potentialVerticesCount}`);
                                        
                                        // Validate by reading some data
                                        const indicesStart = afterArrayOffset + 4;
                                        const verticesStart = afterIndicesOffset + 4;
                                        
                                        console.log("\n  Sample indices:");
                                        const sampleIndices = [];
                                        for (let i = 0; i < Math.min(10, potentialIndicesCount); i++) {
                                            const idx = data.readUInt16LE(indicesStart + (i * 2));
                                            sampleIndices.push(idx);
                                        }
                                        console.log(`    [${sampleIndices.join(', ')}]`);
                                        
                                        console.log("\n  Sample vertices:");
                                        for (let i = 0; i < Math.min(3, potentialVerticesCount); i++) {
                                            const vOffset = verticesStart + (i * 16);
                                            if (vOffset + 16 <= data.length) {
                                                const x = data.readInt16LE(vOffset);
                                                const y = data.readInt16LE(vOffset + 2);
                                                const hNear = data.readInt16LE(vOffset + 6);
                                                console.log(`    Vertex ${i}: x=${x}, y=${y}, hNear=${hNear}`);
                                            }
                                        }
                                        
                                        // Look for render batches
                                        const batchesOffset = verticesStart + (potentialVerticesCount * 16);
                                        if (batchesOffset + 4 <= data.length) {
                                            const batchCount = data.readUInt32LE(batchesOffset);
                                            console.log(`\n  Render batches: ${batchCount}`);
                                            
                                            if (batchCount > 0 && batchCount <= 20) {
                                                for (let i = 0; i < Math.min(batchCount, 5); i++) {
                                                    const bOffset = batchesOffset + 4 + (i * 16);
                                                    if (bOffset + 16 <= data.length) {
                                                        const idxOff = data.readUInt32LE(bOffset);
                                                        const idxCnt = data.readUInt32LE(bOffset + 4);
                                                        const vtxOff = data.readUInt32LE(bOffset + 8);
                                                        const vtxCnt = data.readUInt32LE(bOffset + 12);
                                                        console.log(`    Batch ${i}: idxOff=${idxOff}, idxCnt=${idxCnt}, vtxOff=${vtxOff}, vtxCnt=${vtxCnt}`);
                                                    }
                                                }
                                            }
                                        }
                                        
                                        // Save the working offsets
                                        const result = {
                                            success: true,
                                            geometryOffset: searchOffset,
                                            unk1: unk1,
                                            arrayCount: arrayCount,
                                            indicesOffset: afterArrayOffset,
                                            indicesCount: potentialIndicesCount,
                                            verticesOffset: afterIndicesOffset,
                                            verticesCount: potentialVerticesCount,
                                            sampleIndices: sampleIndices
                                        };
                                        
                                        fs.writeFileSync('./GEOMETRY_FOUND.json', JSON.stringify(result, null, 2));
                                        console.log("\n✅ GEOMETRY STRUCTURE SAVED TO GEOMETRY_FOUND.json");
                                        
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } catch (e) {
                    // Continue searching
                }
            }
            
            if (!found) {
                console.log("❌ Could not locate geometry data with comprehensive search");
            }
        }
    }
}

// Run the parser
if (process.argv.length >= 3) {
    parseMinimalCNK0(process.argv[2]);
} else {
    console.log("Usage: node examine_js_schema.js <cnk0_file>");
}