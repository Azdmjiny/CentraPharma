#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def calc_formal_charge_from_smiles(smiles: str) -> int:
    if not smiles:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(Chem.GetFormalCharge(mol))


def resolve_and_validate_charges(item: dict, smiles: str):
    """
    统一电荷来源：
    1) RDKit 从 SMILES 算 formal charge
    2) 如果 item 里已有 net_charge，则优先采用
    """
    rdkit_formal_charge = calc_formal_charge_from_smiles(smiles)

    net_charge = (
        item.get("net_charge")
        or (item.get("drug_properties") or {}).get("net_charge")
        or (item.get("delivery_system") or {}).get("net_charge")
    )

    if net_charge is None:
        net_charge = rdkit_formal_charge

    try:
        net_charge = int(round(float(net_charge)))
    except Exception:
        net_charge = rdkit_formal_charge

    return rdkit_formal_charge, net_charge


def _maybe_unzip_templates(template_dir: Path):
    if template_dir.is_dir():
        return template_dir

    zip_path = template_dir.with_suffix(".zip")
    if zip_path.exists():
        template_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(template_dir)

        inner = template_dir / "template"
        if inner.is_dir():
            for child in inner.iterdir():
                target = template_dir / child.name
                if not target.exists():
                    child.rename(target)
            try:
                inner.rmdir()
            except Exception:
                pass
        return template_dir

    alt_zip = template_dir.parent / "template.zip"
    if alt_zip.exists() and (not template_dir.exists()):
        template_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(alt_zip, "r") as zf:
            zf.extractall(template_dir)

        inner = template_dir / "template"
        if inner.is_dir():
            for child in inner.iterdir():
                target = template_dir / child.name
                if not target.exists():
                    child.rename(target)
            try:
                inner.rmdir()
            except Exception:
                pass
        return template_dir

    return template_dir


def _win_path_to_posix_guess(s: str) -> str:
    if not s:
        return s
    s = str(s).strip()
    if re.match(r"^[A-Za-z]:\\", s):
        return Path(s).name
    return s.replace("\\", "/")


def _choose_forcefield_plan(material: str):
    import os
    from pathlib import Path

    m = (material or "").upper().strip()
    amberhome = os.environ.get("AMBERHOME", "").strip()

    def _pick_existing(*relative_candidates: str, fallback: str):
        if amberhome:
            base = Path(amberhome) / "dat" / "leap" / "cmd"
            for rel in relative_candidates:
                p = base / rel
                if p.exists():
                    return str(p)
        return fallback

    water_rc = _pick_existing(
        "leaprc.water.tip3p",
        fallback="leaprc.water.tip3p",
    )

    gaff2_rc = _pick_existing(
        "leaprc.gaff2",
        fallback="leaprc.gaff2",
    )

    if m in ("LNP", "LIPOSOME"):
        lipid_rc = _pick_existing(
            "leaprc.lipid17",
            "oldff/leaprc.lipid17",
            "leaprc.lipid21",
            fallback="leaprc.lipid21",
        )
        return {
            "family": "amber",
            "leaprc": [lipid_rc, gaff2_rc, water_rc],
            "small_molecule_ff": "gaff2",
            "charge_method": "am1bcc",
            "notes": "LNP/Liposome lipids via Lipid17 if available, otherwise oldff Lipid17, otherwise Lipid21; drug via GAFF2",
        }

    return {
        "family": "amber",
        "leaprc": [gaff2_rc, water_rc],
        "small_molecule_ff": "gaff2",
        "charge_method": "am1bcc",
        "notes": "Organic/polymer system via GAFF2",
    }



def _calc_formula_from_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def _find_molid_from_single_molecule_csv(result_dir: Path, smiles: str) -> str:
    candidates = [
        result_dir / "single_molecule.csv",
        result_dir / "molecules" / "single_molecule.csv",
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if {"smiles", "mol_id"}.issubset(df.columns):
                row = df.loc[df["smiles"] == smiles]
                if not row.empty:
                    return str(row.iloc[0]["mol_id"])

    for p in result_dir.rglob("single_molecule.csv"):
        try:
            df = pd.read_csv(p)
            if {"smiles", "mol_id"}.issubset(df.columns):
                row = df.loc[df["smiles"] == smiles]
                if not row.empty:
                    return str(row.iloc[0]["mol_id"])
        except Exception:
            continue

    return ""


def build_agent_manifest_from_result_once(result_once_path: Path, template_dir: Path, out_path: Path):
    result = _read_json(result_once_path)

    if isinstance(result, list) and result:
        item = result[0]
    elif isinstance(result, dict):
        item = result
    else:
        raise ValueError("result_once.json 格式不对：需要 dict 或非空 list[dict]")

    smiles = item.get("smiles", "")
    drug_props = item.get("drug_properties", {}) or {}
    delivery = item.get("delivery_system", {}) or {}
    md_metrics = item.get("md_metrics", {}) or {}

    material = delivery.get("material", "PLGA")
    packmol_ok = bool(delivery.get("packmol_ok", False))
    openmm_min_pdb = delivery.get("openmm_min_pdb") or md_metrics.get("openmm_min_pdb") or ""

    packmol_pdb = _win_path_to_posix_guess(delivery.get("packmol_pdb", ""))
    if not packmol_pdb:
        raise FileNotFoundError("result_once.json 中没有 delivery_system.packmol_pdb")

    base_dir = result_once_path.parent
    packmol_pdb_path = Path(packmol_pdb)

    if not packmol_pdb_path.is_absolute():
        guesses = list(base_dir.rglob(packmol_pdb_path.name))
        packmol_pdb_path = guesses[0] if guesses else (base_dir / packmol_pdb_path)

    if not packmol_pdb_path.exists():
        raise FileNotFoundError(f"找不到 PACKMOL 输出 PDB: {packmol_pdb_path}")

    candidate_id = item.get("candidate_id", "CAND_0001")
    best_design_id = item.get("best_design_id", f"{candidate_id}_D001")

    run_dir = packmol_pdb_path.parent
    amber_dir = run_dir / "amber"
    openmm_dir = run_dir / "openmm"
    reports_dir = run_dir / "reports"
    metrics_json = reports_dir / "md_metrics_real.json"

    ff_plan = _choose_forcefield_plan(material)
    mol_id = _find_molid_from_single_molecule_csv(base_dir, smiles)
    formula = _calc_formula_from_smiles(smiles)
    formal_charge, net_charge = resolve_and_validate_charges(item, smiles)

    template_dir = _maybe_unzip_templates(template_dir)
    if not template_dir.exists():
        raise FileNotFoundError(f"template_dir 不存在：{template_dir}")

    manifest = {
        "bundle_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "result_once_json": str(result_once_path),
            "template_dir": str(template_dir),
            "packmol_pdb_detected": str(packmol_pdb_path),
        },
        "global_defaults": {
            "parameterization": {
                "engine": "ambertools",
                "drug_parameter_mode": "runtime_from_smiles",
                "carrier_parameter_mode": "template_prebuilt_preferred",
                "forcefield_plan": ff_plan,
                "neutralize_system": True,
                "ion_strength_molar": 0.15,
            },
            "openmm": {
                "mode": "amber_prmtop_inpcrd",
                "platform_preference": ["CUDA", "OpenCL", "CPU"],
                "constraints": "HBonds",
                "timestep_fs": 2.0,
                "temperature_K": 300.0,
                "pressure_bar": 1.0,
                "minimize_max_iterations": 5000,
                "equilibration": {
                    "nvt_ps": 200.0,
                    "npt_ps": 500.0,
                },
                "production": {
                    "time_ns": 2.0,
                    "report_interval_ps": 10.0,
                },
            },
        },
        "candidates": [
            {
                "candidate_id": candidate_id,
                "best_design_id": best_design_id,
                "packmol_ok": packmol_ok,
                "drug": {
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "formula": formula,
                    "formal_charge_from_smiles": formal_charge,
                    "net_charge": net_charge,
                    "properties": drug_props,
                },
                "carrier": {
                    "material": material,
                    "type": delivery.get("type", "nanoparticle"),
                    "size_nm": float(delivery.get("size_nm", 0.0) or 0.0),
                    "zeta_mv": float(delivery.get("zeta_mv", 0.0) or 0.0),
                    "drug_loading": float(delivery.get("drug_loading", 0.0) or 0.0),
                },
                "system_inputs": {
                    "packmol_pdb": str(packmol_pdb_path),
                    "template_dir": str(template_dir),
                    "material_packmol_spec": str(template_dir / f"{material}_packmol.json"),
                    "parameter_strategy": {
                        "drug": "runtime_from_smiles",
                        "carrier": "template_prebuilt_preferred",
                    },
                },
                "outputs": {
                    "amber": {
                        "work_dir": str(amber_dir),
                        "drug_mol2": str(amber_dir / "drug.mol2"),
                        "drug_frcmod": str(amber_dir / "drug.frcmod"),
                        "system_prmtop": str(amber_dir / "system.prmtop"),
                        "system_inpcrd": str(amber_dir / "system.inpcrd"),
                        "tleap_log": str(amber_dir / "tleap.log"),
                    },
                    "openmm": {
                        "work_dir": str(openmm_dir),
                        "minimized_pdb": str(openmm_dir / "system_openmm_min_real.pdb"),
                        "traj_dcd": str(openmm_dir / "traj.dcd"),
                        "state_xml": str(openmm_dir / "state.xml"),
                        "legacy_openmm_min_pdb": str(openmm_min_pdb) if openmm_min_pdb else "",
                    },
                    "analysis": {
                        "work_dir": str(reports_dir),
                        "metrics_json": str(metrics_json),
                        "metrics_csv": str(reports_dir / "md_metrics_real.csv"),
                    },
                },
            }
        ],
    }

    _write_json(out_path, manifest)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Convert pipeline result_once.json to an agent-ready manifest.")
    ap.add_argument("--result_once", required=True)
    ap.add_argument("--template_dir", default=str(Path(__file__).resolve().parent / "template"))
    ap.add_argument("--out", default="agent_manifest.json")
    args = ap.parse_args()

    out = build_agent_manifest_from_result_once(
        Path(args.result_once),
        Path(args.template_dir),
        Path(args.out),
    )
    print(out)


if __name__ == "__main__":
    main()