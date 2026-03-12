import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "drug_template.pdb": "DRG",
    "drug.pdb": "DRG",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _which_or_raise(exe_name: str):
    path = shutil.which(exe_name)
    if not path:
        raise FileNotFoundError(f"没有找到可执行文件: {exe_name}")
    return path


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, log_file: Optional[Path] = None):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )

    if log_file:
        _ensure_parent(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.write(proc.stdout or "")
            if proc.stderr:
                f.write("\n[STDERR]\n")
                f.write(proc.stderr)
            f.write(f"\n[RETURNCODE] {proc.returncode}\n")

    if proc.returncode != 0:
        raise RuntimeError(
            "命令执行失败:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc


def _formal_charge_from_smiles(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(Chem.GetFormalCharge(mol))


def _write_drug_sdf_from_smiles(smiles: str, output_sdf: Path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法从 SMILES 解析分子: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 20250309
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("RDKit 3D 构象生成失败")

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    except Exception:
        pass

    _ensure_parent(output_sdf)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()


def _parse_mol2_atom_names(mol2_path: Path) -> List[str]:
    atom_names = []
    in_atom_block = False
    with open(mol2_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if s.upper().startswith("@<TRIPOS>BOND"):
                break
            if in_atom_block and s:
                parts = s.split()
                if len(parts) >= 2:
                    atom_names.append(parts[1][:4])
    if not atom_names:
        raise ValueError(f"无法从 mol2 读取 atom names: {mol2_path}")
    return atom_names


def _load_packmol_spec(spec_path: Path) -> dict:
    if not spec_path.exists():
        raise FileNotFoundError(f"找不到 packmol spec: {spec_path}")
    return _read_json(spec_path)


def _collect_unique_component_templates(spec: dict) -> List[str]:
    comps = spec.get("components", [])
    seen = []
    used = set()
    for c in comps:
        pdb = c.get("pdb")
        if pdb and pdb not in used:
            used.add(pdb)
            seen.append(pdb)
    return seen


def _parameterize_template_pdb(
    input_pdb: Path,
    out_mol2: Path,
    out_frcmod: Path,
    residue_name: str,
    log_file: Path,
    net_charge: int = 0,
):
    _which_or_raise("antechamber")
    _which_or_raise("parmchk2")

    _run_cmd(
        [
            "antechamber",
            "-i", str(input_pdb),
            "-fi", "pdb",
            "-o", str(out_mol2),
            "-fo", "mol2",
            "-c", "bcc",
            "-nc", str(int(net_charge)),
            "-at", "gaff2",
            "-rn", residue_name[:3],
            "-pf", "y",
        ],
        cwd=out_mol2.parent,
        log_file=log_file,
    )

    _run_cmd(
        [
            "parmchk2",
            "-i", str(out_mol2),
            "-f", "mol2",
            "-o", str(out_frcmod),
            "-s", "gaff2",
        ],
        cwd=out_frcmod.parent,
        log_file=log_file,
    )

    if not out_mol2.exists():
        raise FileNotFoundError(f"未生成 mol2: {out_mol2}")
    if not out_frcmod.exists():
        raise FileNotFoundError(f"未生成 frcmod: {out_frcmod}")


def _parameterize_drug(
    smiles: str,
    out_sdf: Path,
    out_mol2: Path,
    out_frcmod: Path,
    log_file: Path,
):
    _which_or_raise("antechamber")
    _which_or_raise("parmchk2")

    _write_drug_sdf_from_smiles(smiles, out_sdf)
    formal_charge = _formal_charge_from_smiles(smiles)

    _run_cmd(
        [
            "antechamber",
            "-i", str(out_sdf),
            "-fi", "sdf",
            "-o", str(out_mol2),
            "-fo", "mol2",
            "-c", "bcc",
            "-nc", str(formal_charge),
            "-at", "gaff2",
            "-rn", "DRG",
            "-pf", "y",
        ],
        cwd=out_mol2.parent,
        log_file=log_file,
    )

    _run_cmd(
        [
            "parmchk2",
            "-i", str(out_mol2),
            "-f", "mol2",
            "-o", str(out_frcmod),
            "-s", "gaff2",
        ],
        cwd=out_frcmod.parent,
        log_file=log_file,
    )

    return formal_charge


def _rename_system_pdb_atoms_by_resname_templates(
    input_pdb: Path,
    output_pdb: Path,
    resname_to_mol2: Dict[str, Path],
):
    atom_name_map = {res: _parse_mol2_atom_names(mol2) for res, mol2 in resname_to_mol2.items()}

    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    grouped = {}
    for i, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            key = (resname, chain, resseq)
            grouped.setdefault(key, []).append(i)

    for (resname, chain, resseq), idxs in grouped.items():
        if resname not in atom_name_map:
            continue
        template_names = atom_name_map[resname]
        if len(idxs) != len(template_names):
            raise ValueError(
                f"残基 {resname} chain={chain} resseq={resseq} 原子数 {len(idxs)} "
                f"与模板 mol2 原子数 {len(template_names)} 不一致"
            )
        for j, line_idx in enumerate(idxs):
            old = lines[line_idx]
            new_atom_name = f"{template_names[j]:>4}"
            lines[line_idx] = f"{old[:12]}{new_atom_name}{old[16:]}"

    with open(output_pdb, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _copy_file(src: Path, dst: Path):
    _ensure_parent(dst)
    shutil.copy2(src, dst)
    return dst


def _find_template_param_pair(template_dir: Path, comp_pdb_name: str, resname: str) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    stem = Path(comp_pdb_name).stem
    candidates = [
        (template_dir / f"{resname}.mol2", template_dir / f"{resname}.frcmod", f"resname:{resname}"),
        (template_dir / f"{stem}.mol2", template_dir / f"{stem}.frcmod", f"stem:{stem}"),
    ]
    for mol2_path, frcmod_path, tag in candidates:
        if mol2_path.exists() and frcmod_path.exists():
            return mol2_path, frcmod_path, tag
    return None, None, None


def _prepare_component_params(
    comp_pdb_name: str,
    resname: str,
    template_dir: Path,
    amber_work_dir: Path,
    log_file: Path,
) -> Dict[str, str]:
    src_pdb = template_dir / comp_pdb_name
    if not src_pdb.exists():
        raise FileNotFoundError(f"模板组件不存在: {src_pdb}")

    out_mol2 = amber_work_dir / f"{resname}.mol2"
    out_frcmod = amber_work_dir / f"{resname}.frcmod"

    src_mol2, src_frcmod, source_tag = _find_template_param_pair(template_dir, comp_pdb_name, resname)
    if src_mol2 and src_frcmod:
        _copy_file(src_mol2, out_mol2)
        _copy_file(src_frcmod, out_frcmod)
        return {
            "unit_name": resname,
            "mol2_name": out_mol2.name,
            "frcmod_name": out_frcmod.name,
            "resname": resname,
            "parameter_source": "template_prebuilt",
            "parameter_source_detail": source_tag,
            "source_mol2": str(src_mol2),
            "source_frcmod": str(src_frcmod),
        }

    print(f"[AMBER] template params missing for {comp_pdb_name} -> {resname}; fallback to runtime parameterization")
    _parameterize_template_pdb(
        input_pdb=src_pdb,
        out_mol2=out_mol2,
        out_frcmod=out_frcmod,
        residue_name=resname,
        log_file=log_file,
        net_charge=0,
    )
    return {
        "unit_name": resname,
        "mol2_name": out_mol2.name,
        "frcmod_name": out_frcmod.name,
        "resname": resname,
        "parameter_source": "runtime_from_pdb",
        "parameter_source_detail": str(src_pdb),
        "source_mol2": str(out_mol2),
        "source_frcmod": str(out_frcmod),
    }


def _build_tleap_input_text(
    leaprc_list: List[str],
    component_blocks: List[Dict[str, str]],
    normalized_pdb_name: str,
    out_prmtop_name: str,
    out_inpcrd_name: str,
    out_pdb_name: str = "system_leap.pdb",
):
    lines = []
    for rc in leaprc_list:
        lines.append(f"source {rc}")

    lines.append("")
    for block in component_blocks:
        lines.append(f"loadamberparams {block['frcmod_name']}")
        lines.append(f"{block['unit_name']} = loadmol2 {block['mol2_name']}")
    lines.append("")
    lines.append(f"SYS = loadpdb {normalized_pdb_name}")
    lines.append("check SYS")
    lines.append(f"saveamberparm SYS {out_prmtop_name} {out_inpcrd_name}")
    lines.append(f"savepdb SYS {out_pdb_name}")
    lines.append("quit")
    lines.append("")
    return "\n".join(lines)


def build_amber_system_from_manifest(
    manifest_path: str,
    candidate_index: int = 0,
    overwrite: bool = True,
):
    manifest_file = Path(manifest_path).resolve()
    manifest = _read_json(manifest_file)
    item = manifest["candidates"][candidate_index]

    smiles = _safe_get(item, "drug", "smiles")
    packmol_pdb = Path(_safe_get(item, "system_inputs", "packmol_pdb")).resolve()
    template_dir = Path(_safe_get(item, "system_inputs", "template_dir")).resolve()
    spec_path = Path(_safe_get(item, "system_inputs", "material_packmol_spec")).resolve()

    amber_work_dir = Path(_safe_get(item, "outputs", "amber", "work_dir")).resolve()
    drug_mol2 = Path(_safe_get(item, "outputs", "amber", "drug_mol2")).resolve()
    drug_frcmod = Path(_safe_get(item, "outputs", "amber", "drug_frcmod")).resolve()
    system_prmtop = Path(_safe_get(item, "outputs", "amber", "system_prmtop")).resolve()
    system_inpcrd = Path(_safe_get(item, "outputs", "amber", "system_inpcrd")).resolve()
    tleap_log = Path(_safe_get(item, "outputs", "amber", "tleap_log")).resolve()

    amber_work_dir.mkdir(parents=True, exist_ok=True)

    ff_plan = _safe_get(manifest, "global_defaults", "parameterization", "forcefield_plan", default={})
    leaprc_list = ff_plan.get("leaprc") or ["leaprc.gaff2", "leaprc.water.tip3p"]

    drug_sdf = amber_work_dir / "drug.sdf"
    normalized_packmol_pdb = amber_work_dir / "packmol_system_for_tleap.pdb"
    tleap_in = amber_work_dir / "tleap.in"
    leap_pdb = amber_work_dir / "system_leap.pdb"

    if overwrite:
        for p in [drug_sdf, drug_mol2, drug_frcmod, normalized_packmol_pdb, tleap_in, system_prmtop, system_inpcrd, leap_pdb, tleap_log]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    print(f"[AMBER] material spec={spec_path}")
    spec = _load_packmol_spec(spec_path)
    used_components = _collect_unique_component_templates(spec)

    component_blocks = []
    parameterization_summary = {
        "drug_mode": "runtime_from_smiles",
        "carrier_mode": "template_prebuilt_preferred",
        "components": [],
    }

    print("[AMBER] parameterizing drug from SMILES ...")
    formal_charge = _parameterize_drug(
        smiles=smiles,
        out_sdf=drug_sdf,
        out_mol2=drug_mol2,
        out_frcmod=drug_frcmod,
        log_file=tleap_log,
    )
    drug_block = {
        "unit_name": "DRG",
        "mol2_name": drug_mol2.name,
        "frcmod_name": drug_frcmod.name,
        "resname": "DRG",
        "parameter_source": "runtime_from_smiles",
        "parameter_source_detail": f"formal_charge={formal_charge}",
        "source_mol2": str(drug_mol2),
        "source_frcmod": str(drug_frcmod),
    }
    component_blocks.append(drug_block)
    parameterization_summary["components"].append({
        "component": "drug",
        "resname": "DRG",
        "parameter_source": "runtime_from_smiles",
        "parameter_source_detail": f"formal_charge={formal_charge}",
    })

    component_resname_to_mol2 = {"DRG": drug_mol2}

    for comp_pdb_name in used_components:
        resname = COMPONENT_RESNAME_MAP.get(comp_pdb_name)
        if not resname:
            raise KeyError(f"组件 {comp_pdb_name} 没有配置残基名映射")

        print(f"[AMBER] preparing carrier component {comp_pdb_name} -> {resname}")
        block = _prepare_component_params(
            comp_pdb_name=comp_pdb_name,
            resname=resname,
            template_dir=template_dir,
            amber_work_dir=amber_work_dir,
            log_file=tleap_log,
        )
        component_blocks.append(block)
        component_resname_to_mol2[resname] = amber_work_dir / block["mol2_name"]
        parameterization_summary["components"].append({
            "component": comp_pdb_name,
            "resname": resname,
            "parameter_source": block["parameter_source"],
            "parameter_source_detail": block["parameter_source_detail"],
        })

    print("[AMBER] renaming system PDB atom names by component mol2 templates ...")
    _rename_system_pdb_atoms_by_resname_templates(
        input_pdb=packmol_pdb,
        output_pdb=normalized_packmol_pdb,
        resname_to_mol2=component_resname_to_mol2,
    )

    print("[AMBER] writing tleap.in ...")
    tleap_in.write_text(
        _build_tleap_input_text(
            leaprc_list=leaprc_list,
            component_blocks=component_blocks,
            normalized_pdb_name=normalized_packmol_pdb.name,
            out_prmtop_name=system_prmtop.name,
            out_inpcrd_name=system_inpcrd.name,
            out_pdb_name=leap_pdb.name,
        ),
        encoding="utf-8",
    )

    print("[AMBER] running tleap ...")
    _run_cmd(["tleap", "-f", str(tleap_in.name)], cwd=amber_work_dir, log_file=tleap_log)

    if not system_prmtop.exists():
        raise FileNotFoundError(f"未生成 prmtop: {system_prmtop}")
    if not system_inpcrd.exists():
        raise FileNotFoundError(f"未生成 inpcrd: {system_inpcrd}")

    result = {
        "success": True,
        "used_components": used_components,
        "component_blocks": component_blocks,
        "parameterization": parameterization_summary,
        "normalized_packmol_pdb": str(normalized_packmol_pdb),
        "system_prmtop": str(system_prmtop),
        "system_inpcrd": str(system_inpcrd),
        "tleap_in": str(tleap_in),
        "tleap_log": str(tleap_log),
    }

    item.setdefault("build", {})
    item["build"]["ambertools"] = result
    _write_json(manifest_file, manifest)
    _write_json(amber_work_dir / "amber_build_summary.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--candidate-index", type=int, default=0)
    args = ap.parse_args()

    result = build_amber_system_from_manifest(
        manifest_path=args.manifest,
        candidate_index=args.candidate_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
