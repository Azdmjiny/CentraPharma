这里需要放 PACKMOL 递送体系模板文件。

每种 material 需要：
- <MATERIAL>_packmol.json   (高级 packmol 规格，指定 components 列表)
- 以及 json 里引用到的各组分 PDB 文件（例如 DSPC.pdb、chol.pdb 等）

例如：
- LNP_packmol.json + (ionizable_lipid.pdb / DSPC.pdb / cholesterol.pdb / PEG_lipid.pdb ...)
- LIPOSOME_packmol.json + (...)
- NLC_packmol.json + (...)
- PLGA_packmol.json + (...)

注意：当前 structure_packmol.py 已禁用旧的 <MATERIAL>_shell.pdb 简单回退模式；
必须提供 _packmol.json。
