@echo off
for %%S in (cnk1compressed\*.cnk1) do cnkdec d %%S cnk1decompressed\%%~nxS 
echo ALL DONE!
pause