I included all of the requirements kind of pre-installed and standalone for ease (and it was kind of a mess to begin with, but for this project I think it is best to have everything included and is not normal), keep using my instructions (if you have an issue you may need to follow the original pydmod installation instructions, or just install missing dependencies, but you should not need to)

1) ps2ls:

Start at open pydmod\ps2ls_1_2_0_133\ps2ls.exe > select all .pack and extract the the raw files to a single directory, ie M:\H1Z1_assets (or wherever you like, will be several GB)
the initial .pack files are in H1EMU\Resources\Assets 

2) Then take and decompress .cnk1 using modified forgelight-chunk:

grab and MOVE all the .cnk1 from the extracted .pack files folder and bring them into pydmod\forgelight-chunk\cnk1compressed folder, then run pydmod\forgelight-chunk\decompressallcnk1.bat which will extract to pydmod\forgelight-chunk\cnk1decompressed
Don't store the original compressed .cnk1 in the H1Z1_assets folder (just leave them in pydmod\forgelight-chunk\cnk1compressed folder), 
put the NEW uncompressed .cnk1 into the unpacked H1Z1_assets folder (note these will have the exact same filenames as the originals, just larger)


3) Optional edit of dme_loader.py:

pydmod\dme_loader\dme_loader.py is the screener for which .adr files get included or excluded in the final .glb export from zone_converter_kit.py
things like item spawns, zombies, trash, certain props, etc are all not needed for navmesh.  You can add or remove things to keep.  This already has a decent setup already, but allows for customization.  Can keep unchanged if you like.

pydmod\conversion_logs will give an output .txt file of which .adr were kept or skipped, among some other stats, once zone_converter_kit.py is run

4) zone_converter_kit.py:

navigate to your pydmod directory in cmd, powershell or other
cd M:\yourdirectorynaming\pydmod
Activate the virtual environment:
& ./venv/bin/Activate.ps1
or
venv\Scripts\activate
or
.venv\Scripts\activate

how to run zone_converter_kit.py (from your pydmod directory, cmd or powershell or other):

python zone_converter_kit.py "zonefile.zone" "outputpath.glb" --asset-dir "assetpath" options

for options: -a is actors (.adr) enabled, -t is terrain (cnk1) enabled, -v verbose logging, -f format (glb or gltf)

FOR EXAMPLE (in my case):
in powershell:
cd M:\H1_Tool_Projects\pydmod
venv\Scripts\activate

then use the following command for a bounding box x1 z1 x2 z2:
python zone_converter_kit.py "Z1.zone" "M:\NavMesh_Project\TestOutput.glb" --asset-dir "M:\H1Z1_assets" -f glb -v -a -t -b 512 -512 1024 0

or if you want the entire zone file:
python zone_converter_kit.py "Z1.zone" "M:\NavMesh_Project\TestOutput.glb" --asset-dir "M:\H1Z1_assets" -f glb -v -a -t

for more explanation (should not need) visit original pydmod https://github.com/ryanjsims/pydmod (not all original functions are in my version as they are not needed)

5) Open .glb file in Blender

edit your heart out, then export as .obj
Default Settings, then make sure
Foward Axis -Z
Up Axis Y
Triangulated Mesh is checked

6) Move onto Recast and Detour to load the .obj and create .bin (not included in this repo)

You can check my other repos for my version of RecastDemo 64bit, or use the original 32bit



## Third-Party Code

This project includes modified portions of the following third-party libraries:

### ps2ls
*   **Original Source:** [https://github.com/psemu/ps2ls](https://github.com/psemu/ps2ls)
*   **License:** MIT License
*   The original license file is included in the `ps2ls` directory.

### forgelight-chunk
*   **Original Source:** [https://github.com/psemu/forgelight-chunk](https://github.com/psemu/forgelight-chunk)
*   **License:** MIT License
*   The original license file is included in the `forgelight-chunk` directory.

The rest should be credited within pydmod, if I missed anything my apologies, not my intent.

We are grateful to the original authors for their work. The code in these directories has been modified from its original state to suit the needs of this project.