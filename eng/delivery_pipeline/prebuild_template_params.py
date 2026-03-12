#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

COMPONENT_RESNAME_MAP = {
    "PLGA50_50_20mer.pdb": "PLG",
    "PEG5k_ligand_stub.pdb": "PEG",
    "CHOL.pdb": "CHL",
    "COMPRITOL888ATO.pdb": "COM",
    "DMG_PEG2000.pdb": "DPG",
    "DSPC.pdb": "DSP",
    "DSPE_PEG2000.pdb": "DSG",
    "HSPC.pdb": "HSC",
    "MC3.pdb": "MC3",
    "MIGLYOL812.pdb": "MIG",
    "TWEEN80.pdb": "TWE",
}

DEFAULT_TARGETS = [
    "PLGA50_50_20mer.pdb",
    "PEG5k_ligand_stub.pdb",
    "MC3.pdb",
    "DSPC.pdb",
    "HSPC.pdb",
    "CHOL.pdb",
    "DMG_PEG2000.pdb",
    "DSPE_PEG2000.pdb",
    "COMPRITOL888ATO.pdb",
    "MIGLYOL812.pdb",
    "TWEEN80.pdb",
]

# 这里只保留“你已经实际上传/下载过且对应关系较明确”的候选文件名。
# 模糊或不确定的映射不写进来，避免把主观猜测编码进脚本。
KNOWN_SDF_CANDIDATES = {
    "CHL": ["CHOL.sdf", "Conformer3D_COMPOUND_CID_5997.sdf"],
    "COM": ["COMPRITOL888ATO.sdf", "Structure2D_COMPOUND_CID_62726.sdf"],
    "DPG": ["DMG_PEG2000.sdf", "Structure2D_COMPOUND_CID_10257450.sdf"],
    "DSG": ["DSPE_PEG2000.sdf", "Structure2D_COMPOUND_CID_406952.sdf"],
    "MC3": ["MC3.sdf", "Structure2D_COMPOUND_CID_49785164.sdf"],
    "MIG": ["MIGLYOL812.sdf", "Structure2D_COMPOUND_CID_93356.sdf"],
    "DSP": ["DSPC.sdf", "Structure2D_COMPOUND_CID_94190.sdf"],
    "PLG": ["PLGA50_50_20mer.sdf"],
    "PEG": ["PEG5k_ligand_stub.sdf"],
    # HSPC / TWEEN80 如果没有同名 sdf，则交给用户显式提供，不在这里硬猜。
}


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_cmd(cmd, cwd=None, log_file=None, timeout=300):
    print(f"[{ts()}] START: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    start = time.time()
    lines = []

    try:
        while True:
            line = proc.stdout.readline()
            if line:
                line = line.rstrip("\n")
                lines.append(line)
                if log_file:
                    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")

            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    for extra in rest.splitlines():
                        lines.append(extra)
                        if log_file:
                            with open(log_file, "a", encoding="utf-8") as f:
                                f.write(extra + "\n")
                break

            elapsed = time.time() - start
            if elapsed > timeout:
                proc.kill()
                raise TimeoutError(f"命令超时（>{timeout}s）:\n{' '.join(cmd)}")

        rc = proc.returncode
        elapsed = time.time() - start
        print(f"[{ts()}] END ({elapsed:.1f}s, rc={rc}): {' '.join(cmd)}", flush=True)

        if rc != 0:
            raise RuntimeError(
                "命令执行失败:\n"
                f"{' '.join(cmd)}\n\n"
                "输出尾部:\n" + "\n".join(lines[-80:])
            )

        return "\n".join(lines)

    except Exception:
        elapsed = time.time() - start
        print(f"[{ts()}] FAIL ({elapsed:.1f}s): {' '.join(cmd)}", flush=True)
        raise


def ensure_exe(name: str):
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"未找到可执行文件: {name}")
    return path


def force_pdb_resname(input_pdb: Path, output_pdb: Path, residue_name: str):
    residue_name = residue_name[:3]
    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_pdb, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
                line = f"{line[:17]}{residue_name:>3}{line[20:]}"
            fout.write(line)


def _load_first_valid_sdf_mol(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def _formal_charge_from_mol(mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _formal_charge_from_sdf(sdf_path: Path) -> int:
    mol = _load_first_valid_sdf_mol(sdf_path)
    if mol is None:
        raise ValueError(f"无法从 SDF 读取分子以计算净电荷: {sdf_path}")
    return _formal_charge_from_mol(mol)


def _has_3d_coords(mol) -> bool:
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    if not conf.Is3D():
        return False
    zs = [float(conf.GetAtomPosition(i).z) for i in range(mol.GetNumAtoms())]
    if not zs:
        return False
    return (max(zs) - min(zs)) > 1e-3


def _build_reasonable_3d_from_sdf(input_sdf: Path, output_sdf: Path, residue_name: str, seed: int = 20250309):
    mol = _load_first_valid_sdf_mol(input_sdf)
    if mol is None:
        raise ValueError(f"无法从 SDF 读取分子: {input_sdf}")

    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"SDF 分子为空: {input_sdf}")

    base_mol = Chem.Mol(mol)

    if _has_3d_coords(base_mol):
        mol3d = Chem.AddHs(base_mol, addCoords=True)
    else:
        mol3d = Chem.AddHs(base_mol)
        mol3d.RemoveAllConformers()
        success = False

        for embed_mode in ("ETKDGv3", "ETKDG", "RANDOM"):
            try:
                if embed_mode == "ETKDGv3":
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    res = AllChem.EmbedMolecule(mol3d, params)
                elif embed_mode == "ETKDG":
                    params = AllChem.ETKDG()
                    params.randomSeed = seed
                    res = AllChem.EmbedMolecule(mol3d, params)
                else:
                    res = AllChem.EmbedMolecule(mol3d, randomSeed=seed, useRandomCoords=True)
                if res == 0:
                    success = True
                    break
            except Exception:
                pass

        if not success:
            raise RuntimeError(f"RDKit 3D 构象生成失败: {input_sdf.name}")

    # 尽量优化，但优化失败不直接判死刑；真正的硬判据是是否有有效 3D 坐标。
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol3d, mmffVariant="MMFF94")
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol3d, mmffVariant="MMFF94", maxIters=2000)
        else:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=2000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=2000)
        except Exception:
            pass

    if not _has_3d_coords(mol3d):
        raise RuntimeError(f"3D 构象生成后仍无有效三维坐标: {input_sdf.name}")

    if residue_name:
        mol3d.SetProp("_Name", residue_name[:3].upper())

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol3d)
    writer.close()


def _find_matching_sdf(template_pdb: Path, template_dir: Path, sdf_dir: Optional[Path], residue_name: str) -> Optional[Path]:
    search_dirs = []
    if sdf_dir is not None:
        search_dirs.append(sdf_dir)
    search_dirs.append(template_dir)

    candidates = [
        f"{template_pdb.stem}.sdf",
        f"{residue_name}.sdf",
    ]
    candidates.extend(KNOWN_SDF_CANDIDATES.get(residue_name, []))

    seen = set()
    for d in search_dirs:
        if d is None or (not d.exists()):
            continue
        for name in candidates:
            p = d / name
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.exists():
                return p
    return None


def parameterize_template(
    template_pdb: Path,
    template_dir: Path,
    residue_name: str,
    overwrite: bool = False,
    sdf_dir: Optional[Path] = None,
    prefer_sdf: bool = True,
    fallback_net_charge: int = 0,
    charge_method: str = "gas",
):
    ensure_exe("antechamber")
    ensure_exe("parmchk2")

    residue_name = residue_name[:3].upper()
    stem = template_pdb.stem

    fixed_pdb = template_dir / f"{stem}.{residue_name}.pdb"
    fixed_sdf = template_dir / f"{stem}.{residue_name}.3d.sdf"
    out_mol2 = template_dir / f"{residue_name}.mol2"
    out_frcmod = template_dir / f"{residue_name}.frcmod"
    out_mol2_alias = template_dir / f"{stem}.mol2"
    out_frcmod_alias = template_dir / f"{stem}.frcmod"
    log_file = template_dir / f"{stem}.prebuild.log"

    if (not overwrite) and out_mol2.exists() and out_frcmod.exists():
        print(f"[SKIP] {template_pdb.name} -> {residue_name} 已存在预制参数")
        return

    if overwrite and log_file.exists():
        log_file.unlink()

    input_path = None
    input_format = None
    effective_net_charge = int(fallback_net_charge)

    source_sdf = None
    if prefer_sdf:
        source_sdf = _find_matching_sdf(template_pdb, template_dir, sdf_dir, residue_name)

    if source_sdf is not None:
        print(f"[3D] 使用 SDF 生成合理 3D: {source_sdf.name} -> {fixed_sdf.name}")
        _build_reasonable_3d_from_sdf(source_sdf, fixed_sdf, residue_name=residue_name)
        input_path = fixed_sdf
        input_format = "sdf"
        try:
            effective_net_charge = _formal_charge_from_sdf(source_sdf)
            print(f"[CHARGE] 从 SDF 自动读取净电荷: {source_sdf.name} -> {effective_net_charge}")
        except Exception as e:
            print(f"[CHARGE-WARN] 无法从 SDF 自动读取净电荷，回退到 --fallback-net-charge={effective_net_charge}: {e}")
    else:
        print(f"[WARN] 没找到 {template_pdb.name} 对应的 SDF，回退到 PDB 直接参数化")
        force_pdb_resname(template_pdb, fixed_pdb, residue_name)
        input_path = fixed_pdb
        input_format = "pdb"
        print(f"[CHARGE] 当前输入为 PDB，使用回退净电荷: {effective_net_charge}")

    print(f"[BUILD] {template_pdb.name} -> {residue_name} (input={input_format}, charge_method={charge_method}, net_charge={effective_net_charge})")

    run_cmd([
        "antechamber",
        "-i", str(input_path),
        "-fi", input_format,
        "-o", str(out_mol2),
        "-fo", "mol2",
        "-c", charge_method,
        "-nc", str(int(effective_net_charge)),
        "-at", "gaff2",
        "-rn", residue_name,
        "-pf", "y",
    ], cwd=template_dir, log_file=log_file, timeout=1800)

    run_cmd([
        "parmchk2",
        "-i", str(out_mol2),
        "-f", "mol2",
        "-o", str(out_frcmod),
        "-s", "gaff2",
    ], cwd=template_dir, log_file=log_file, timeout=600)

    if out_mol2.resolve() != out_mol2_alias.resolve():
        shutil.copy2(out_mol2, out_mol2_alias)
    else:
        print(f"[ALIAS-SKIP] {out_mol2.name} 与别名文件相同，跳过复制")

    if out_frcmod.resolve() != out_frcmod_alias.resolve():
        shutil.copy2(out_frcmod, out_frcmod_alias)
    else:
        print(f"[ALIAS-SKIP] {out_frcmod.name} 与别名文件相同，跳过复制")

    print(f"[DONE] {out_mol2.name}, {out_frcmod.name}")


def main():
    ap = argparse.ArgumentParser(description="为 delivery_pipeline/template 里的模板预制 Amber 参数文件")
    ap.add_argument(
        "--template-dir",
        default="/mnt/d/桌面/AIagent/eng/delivery_pipeline/template",
        help="模板目录"
    )
    ap.add_argument(
        "--sdf-dir",
        default=None,
        help="可选：SDF 所在目录。若不给，则先在 template-dir 下找，再按已知文件名匹配。"
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只处理这些模板文件名，例如 --only MC3.pdb DSPC.pdb"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 mol2/frcmod"
    )
    ap.add_argument(
        "--prefer-pdb",
        action="store_true",
        help="即使找到 SDF 也不用，强制走旧的 PDB 参数化路径。"
    )
    ap.add_argument(
        "--fallback-net-charge",
        type=int,
        default=0,
        help="只有没有可用 SDF 时才使用的回退净电荷。默认 0。"
    )
    ap.add_argument(
        "--charge-method",
        choices=["gas", "bcc"],
        default="gas",
        help="antechamber 电荷方法。默认 gas；若你明确要更慢的 AM1-BCC，可手动改成 bcc。"
    )
    args = ap.parse_args()

    template_dir = Path(args.template_dir).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"template_dir 不存在: {template_dir}")

    sdf_dir = Path(args.sdf_dir).resolve() if args.sdf_dir else None
    targets = args.only if args.only else DEFAULT_TARGETS

    for name in targets:
        pdb_path = template_dir / name
        if not pdb_path.exists():
            print(f"[MISS] 模板不存在: {pdb_path}")
            continue

        residue_name = COMPONENT_RESNAME_MAP.get(name)
        if not residue_name:
            print(f"[MISS] 未配置残基名映射: {name}")
            continue

        parameterize_template(
            template_pdb=pdb_path,
            template_dir=template_dir,
            residue_name=residue_name,
            overwrite=args.overwrite,
            sdf_dir=sdf_dir,
            prefer_sdf=(not args.prefer_pdb),
            fallback_net_charge=args.fallback_net_charge,
            charge_method=args.charge_method,
        )


if __name__ == "__main__":
    main()
