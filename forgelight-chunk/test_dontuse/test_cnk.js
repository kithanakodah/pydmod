#!/usr/bin/env node

// Simple test script - save as test_cnk.js in the same directory as cnk.js

try {
    const Chunk = require('./cnk.js');
    const fs = require('fs');
    
    if (process.argv.length < 3) {
        console.log("Usage: node test_cnk.js <cnk0_file>");
        process.exit(1);
    }
    
    const cnkFile = process.argv[2];
    
    if (!fs.existsSync(cnkFile)) {
        console.log("File not found:", cnkFile);
        process.exit(1);
    }
    
    console.log("🔍 Loading CNK0 file:", cnkFile);
    console.log("");
    
    // Use the working JavaScript parser
    const chunk = Chunk.read(cnkFile);
    
    console.log("=== CNK0 STRUCTURE ===");
    console.log("Type:", chunk.type);
    console.log("Version:", chunk.version);
    console.log("Available properties:", Object.keys(chunk));
    console.log("");
    
    // Key geometry data
    console.log("=== GEOMETRY DATA ===");
    console.log("Tiles:", chunk.tiles ? chunk.tiles.length : 0);
    console.log("Vertices:", chunk.vertices ? chunk.vertices.length : 0);
    console.log("Indices:", chunk.indices ? chunk.indices.length : 0);
    console.log("Render Batches:", chunk.renderBatches ? chunk.renderBatches.length : 0);
    console.log("");
    
    // Show sample data
    if (chunk.vertices && chunk.vertices.length > 0) {
        console.log("=== SAMPLE VERTEX ===");
        console.log("First vertex:", chunk.vertices[0]);
        console.log("Vertex structure:", Object.keys(chunk.vertices[0]));
        console.log("");
    }
    
    if (chunk.renderBatches && chunk.renderBatches.length > 0) {
        console.log("=== SAMPLE RENDER BATCH ===");
        console.log("First render batch:", chunk.renderBatches[0]);
        console.log("Render batch structure:", Object.keys(chunk.renderBatches[0]));
        console.log("");
    }
    
    if (chunk.indices && chunk.indices.length > 0) {
        console.log("=== SAMPLE INDICES ===");
        console.log("First 10 indices:", chunk.indices.slice(0, 10));
        console.log("");
    }
    
    // Create a detailed JSON structure file
    const outputFile = cnkFile.replace('.cnk0', '_STRUCTURE.json').replace(/.*[\\\/]/, './');
    
    // Create clean version for JSON export
    const structure = {
        fileInfo: {
            type: chunk.type,
            version: chunk.version,
            properties: Object.keys(chunk)
        },
        geometry: {
            vertexCount: chunk.vertices ? chunk.vertices.length : 0,
            indexCount: chunk.indices ? chunk.indices.length : 0,
            renderBatchCount: chunk.renderBatches ? chunk.renderBatches.length : 0,
            tileCount: chunk.tiles ? chunk.tiles.length : 0
        },
        sampleVertex: chunk.vertices && chunk.vertices.length > 0 ? chunk.vertices[0] : null,
        sampleRenderBatch: chunk.renderBatches && chunk.renderBatches.length > 0 ? chunk.renderBatches[0] : null,
        sampleIndices: chunk.indices ? chunk.indices.slice(0, 20) : [],
        allRenderBatches: chunk.renderBatches || [],
        // Include first few vertices for analysis
        firstVertices: chunk.vertices ? chunk.vertices.slice(0, 10) : []
    };
    
    fs.writeFileSync(outputFile, JSON.stringify(structure, null, 2));
    
    console.log("✅ SUCCESS!");
    console.log(`📄 Detailed structure saved to: ${outputFile}`);
    console.log("");
    console.log("🎯 This shows the EXACT data structure we need for Python!");
    
} catch (error) {
    console.error("❌ Error:", error.message);
    console.log("Make sure cnk.js and dataschema.js are in the same directory");
    console.log("Error details:", error.stack);
}