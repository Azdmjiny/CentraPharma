import json
import math
import os
from pathlib import Path

from .admet_ingest import load_admet_candidates
from .structure_packmol import enumerate_designs, run_packmol
from .packmol_to_result import build_agent_manifest_from_result_once
from .amber_builder import build_amber_system_from_manifest
from .openmm_minimizer import run_openmm_minimization
from .scoring import score_designs


DEFAULT_CONFIG = {
    "max_candidates": 3,
    "structures_per_candidate": 4,
    "work_root": "output/delivery_runs",
    "packmol_bin": "packmol",
    "component_dir": "delivery_pipeline/template",
    "enable_openmm": True,
    "analysis_json_name": "md_metrics_real.json",
    "openmm_max_iterations": 5000,
    "weights": {
        "material": 0.30,
        "structure": 0.30,
        "md": 0.30,
        "qed": 0.10,
    },
}


def _eng_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_under_eng(p: str | os.PathLike | None) -> str | None:
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return str((_eng_root() / p).resolve())


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _write_json(path: str | Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _infer_delivery_type(material: str, strategy: str) -> str:
    material_u = str(material or "").upper()
    strategy_l = str(strategy or "").lower()

    if "liposome" in strategy_l or material_u == "LIPOSOME":
        return "liposome"
    if material_u == "LNP":
        return "lipid_nanoparticle"
    if material_u == "NLC":
        return "nanostructured_lipid_carrier"
    return "nanoparticle"




def _read_pdb_atoms(pdb_path: str | Path):
    atoms = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except Exception:
                    continue
                atoms.append({
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resseq": line[22:26].strip(),
                    "xyz": (x, y, z),
                })
    if not atoms:
        raise ValueError(f"无法从 PDB 读取原子坐标: {pdb_path}")
    return atoms


def _mean_xyz(points):
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    sx = sy = sz = 0.0
    for x, y, z in points:
        sx += x
        sy += y
        sz += z
    return (sx / n, sy / n, sz / n)


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _rmsd_same_order(points_a, points_b):
    n = len(points_a)
    if n == 0 or n != len(points_b):
        return None
    s = 0.0
    for pa, pb in zip(points_a, points_b):
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        dz = pa[2] - pb[2]
        s += dx * dx + dy * dy + dz * dz
    return math.sqrt(s / n)


def _radius_of_gyration(points):
    if not points:
        return None
    c = _mean_xyz(points)
    s = 0.0
    for p in points:
        d = _dist(p, c)
        s += d * d
    return math.sqrt(s / len(points))


def _group_residue_centers(atoms, target_resname: str):
    grouped = {}
    for atom in atoms:
        if atom["resname"] != target_resname:
            continue
        key = (atom["resname"], atom["chain"], atom["resseq"])
        grouped.setdefault(key, []).append(atom["xyz"])
    return [_mean_xyz(pts) for pts in grouped.values()]


def _compute_result_check(packmol_pdb: str | Path, minimized_pdb: str | Path):
    before_atoms = _read_pdb_atoms(packmol_pdb)
    after_atoms = _read_pdb_atoms(minimized_pdb)

    before_xyz = [a["xyz"] for a in before_atoms]
    after_xyz = [a["xyz"] for a in after_atoms]

    atom_count_before = len(before_xyz)
    atom_count_after = len(after_xyz)
    same_atom_count = (atom_count_before == atom_count_after)

    all_rmsd_A = _rmsd_same_order(before_xyz, after_xyz) if same_atom_count else None

    disps = []
    if same_atom_count:
        for pa, pb in zip(before_xyz, after_xyz):
            disps.append(_dist(pa, pb))

    center_before = _mean_xyz(before_xyz)
    center_after = _mean_xyz(after_xyz)
    center_shift_A = _dist(center_before, center_after)

    rg_before = _radius_of_gyration(before_xyz)
    rg_after = _radius_of_gyration(after_xyz)
    rg_delta_A = None if (rg_before is None or rg_after is None) else (rg_after - rg_before)

    drug_centers_before = _group_residue_centers(before_atoms, "DRG")
    drug_centers_after = _group_residue_centers(after_atoms, "DRG")
    drug_radius_before = [_dist(c, center_before) for c in drug_centers_before]
    drug_radius_after = [_dist(c, center_after) for c in drug_centers_after]

    drug_radius_mean_before_A = None if not drug_radius_before else sum(drug_radius_before) / len(drug_radius_before)
    drug_radius_mean_after_A = None if not drug_radius_after else sum(drug_radius_after) / len(drug_radius_after)
    drug_radius_shift_mean_A = None
    if drug_radius_mean_before_A is not None and drug_radius_mean_after_A is not None:
        drug_radius_shift_mean_A = drug_radius_mean_after_A - drug_radius_mean_before_A

    score = 1.0
    if not same_atom_count:
        score -= 0.60
    if all_rmsd_A is not None:
        score -= min(0.45, all_rmsd_A / 4.0 * 0.45)
    if rg_delta_A is not None:
        score -= min(0.20, abs(rg_delta_A) / 8.0 * 0.20)
    if center_shift_A is not None:
        score -= min(0.10, center_shift_A / 5.0 * 0.10)
    max_disp = None
    if disps:
        max_disp = max(disps)
        score -= min(0.15, max_disp / 10.0 * 0.15)

    return {
        "atom_count_before": atom_count_before,
        "atom_count_after": atom_count_after,
        "same_atom_count": bool(same_atom_count),
        "all_atom_rmsd_A": None if all_rmsd_A is None else float(all_rmsd_A),
        "mean_displacement_A": None if not disps else float(sum(disps) / len(disps)),
        "max_displacement_A": None if max_disp is None else float(max_disp),
        "center_shift_A": float(center_shift_A),
        "radius_of_gyration_before_A": None if rg_before is None else float(rg_before),
        "radius_of_gyration_after_A": None if rg_after is None else float(rg_after),
        "radius_of_gyration_delta_A": None if rg_delta_A is None else float(rg_delta_A),
        "drug_residue_count_before": len(drug_centers_before),
        "drug_residue_count_after": len(drug_centers_after),
        "drug_radius_mean_before_A": None if drug_radius_mean_before_A is None else float(drug_radius_mean_before_A),
        "drug_radius_mean_after_A": None if drug_radius_mean_after_A is None else float(drug_radius_mean_after_A),
        "drug_radius_mean_shift_A": None if drug_radius_shift_mean_A is None else float(drug_radius_shift_mean_A),
        "stability_index": max(0.0, min(1.0, float(score))),
        "interpretation": "该指标仅表示最小化前后几何变化是否温和，不等同于真实动力学稳定性。",
    }

def _build_single_output(best_item: dict) -> dict:
    design = best_item.get("design", {}) or {}
    desc = design.get("descriptors", {}) or {}
    admet = design.get("admet", {}) or {}
    md_metrics = design.get("md_metrics", {}) or {}
    score_breakdown = best_item.get("score_breakdown", {}) or {}

    strategy = str(design.get("strategy", "") or "")
    material = design.get("material", None)
    delivery_type = _infer_delivery_type(material, strategy)

    advantages = []
    qed_val = _safe_float(desc.get("QED", admet.get("QED")), None)
    if qed_val is not None:
        if qed_val >= 0.67:
            advantages.append("结构参数接近最优窗口")
        elif qed_val >= 0.50:
            advantages.append("结构参数整体较平衡")

    if design.get("packmol_ok", False):
        advantages.append("已完成PACKMOL体系构建")
    else:
        advantages.append("已完成递送系统结构设计")

    md_mode = str(md_metrics.get("mode", "") or "").lower()
    stability_index = _safe_float(md_metrics.get("stability_index"), None)
    if stability_index is not None:
        advantages.append(f"已完成真实最小化与指标计算({md_mode})，稳定性指标={stability_index:.3f}")
    elif md_mode in {"skipped", "unavailable"}:
        advantages.append("OpenMM 暂未接通，当前仅保留结构构建结果")
    elif md_mode == "failed":
        advantages.append("真实最小化未完成，已保留错误信息")
    else:
        advantages.append("已完成真实最小化")

    return {
        "category": "drug_delivery_design",
        "smiles": design.get("smiles"),
        "drug_properties": {
            "MW": _safe_float(desc.get("MW")),
            "logP": _safe_float(desc.get("logP")),
            "tPSA": _safe_float(desc.get("tPSA")),
            "HBA": _safe_float(desc.get("HBA")),
            "HBD": _safe_float(desc.get("HBD")),
            "RotB": _safe_float(desc.get("RotB")),
            "QED": _safe_float(desc.get("QED", admet.get("QED"))),
            "BBB": _safe_float(admet.get("BBB")),
            "hERG": _safe_float(admet.get("hERG")),
            "AMES": _safe_float(admet.get("AMES")),
            "Caco2": _safe_float(admet.get("Caco2")),
            "pKa": admet.get("pKa"),
            "logD74": admet.get("logD74"),
            "Tm": admet.get("Tm"),
            "Solubility": admet.get("Solubility"),
        },
        "delivery_system": {
            "type": delivery_type,
            "material": design.get("material"),
            "targeting_ligand": design.get("targeting_ligand"),
            "size_nm": _safe_float(design.get("size_nm")),
            "zeta_mv": _safe_float(design.get("zeta_mv")),
            "drug_loading": _safe_float(design.get("drug_loading")),
            "packmol_ok": bool(design.get("packmol_ok", False)),
            "packmol_pdb": design.get("packmol_pdb"),
            "openmm_min_pdb": md_metrics.get("openmm_min_pdb"),
            "manifest_path": design.get("manifest_path"),
        },
        "md_metrics": md_metrics,
        "bbb_strategy": {
            "method": design.get("bbb_method"),
            "ligand": design.get("targeting_ligand"),
        },
        "advantages": advantages,
        "score": {
            "total": _safe_float(best_item.get("score_total")),
            "breakdown": {
                "S_material": _safe_float(score_breakdown.get("S_material")),
                "S_structure": _safe_float(score_breakdown.get("S_structure")),
                "S_md": _safe_float(score_breakdown.get("S_md")),
                "S_qed": _safe_float(score_breakdown.get("S_qed")),
            },
        },
        "best_design_id": design.get("design_id"),
        "candidate_id": design.get("candidate_id"),
    }


def _build_outputs_per_molecule(scored: list) -> list:
    best_by_candidate = {}
    for item in scored:
        design = item.get("design", {}) or {}
        candidate_id = design.get("candidate_id", "UNKNOWN")
        score_total = _safe_float(item.get("score_total"), -1e18)
        old = best_by_candidate.get(candidate_id)
        if old is None or score_total > _safe_float(old.get("score_total"), -1e18):
            best_by_candidate[candidate_id] = item

    outputs = [_build_single_output(best_item) for best_item in best_by_candidate.values()]
    outputs.sort(key=lambda x: _safe_float(x.get("score", {}).get("total"), -1e18), reverse=True)
    return outputs


def _build_temp_result_for_manifest(design: dict) -> dict:
    descriptors = design.get("descriptors", {}) or {}
    admet = design.get("admet", {}) or {}
    material = design.get("material")
    strategy = design.get("strategy", "")
    delivery_type = _infer_delivery_type(material, strategy)

    return {
        "category": "drug_delivery_design",
        "smiles": design.get("smiles"),
        "drug_properties": {
            "MW": _safe_float(descriptors.get("MW")),
            "logP": _safe_float(descriptors.get("logP")),
            "tPSA": _safe_float(descriptors.get("tPSA")),
            "HBA": _safe_float(descriptors.get("HBA")),
            "HBD": _safe_float(descriptors.get("HBD")),
            "RotB": _safe_float(descriptors.get("RotB")),
            "QED": _safe_float(descriptors.get("QED", admet.get("QED"))),
            "BBB": _safe_float(admet.get("BBB")),
            "hERG": _safe_float(admet.get("hERG")),
            "AMES": _safe_float(admet.get("AMES")),
            "Caco2": _safe_float(admet.get("Caco2")),
            "pKa": admet.get("pKa"),
            "logD74": admet.get("logD74"),
            "Tm": admet.get("Tm"),
            "Solubility": admet.get("Solubility"),
        },
        "delivery_system": {
            "type": delivery_type,
            "material": material,
            "targeting_ligand": design.get("targeting_ligand"),
            "size_nm": _safe_float(design.get("size_nm")),
            "zeta_mv": _safe_float(design.get("zeta_mv")),
            "drug_loading": _safe_float(design.get("drug_loading")),
            "packmol_ok": bool(design.get("packmol_ok", False)),
            "packmol_pdb": design.get("packmol_pdb"),
            "openmm_min_pdb": None,
        },
        "md_metrics": {},
        "best_design_id": design.get("design_id"),
        "candidate_id": design.get("candidate_id"),
    }


def _run_real_md_from_manifest(manifest_path: str, max_iterations: int = 5000) -> dict:
    manifest = _read_json(manifest_path)
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError("manifest 中没有 candidates")

    item = candidates[0]
    amber_outputs = ((item.get("outputs") or {}).get("amber") or {})
    openmm_outputs = ((item.get("outputs") or {}).get("openmm") or {})
    analysis_outputs = ((item.get("outputs") or {}).get("analysis") or {})
    openmm_cfg = ((manifest.get("global_defaults") or {}).get("openmm") or {})

    prmtop_path = amber_outputs.get("system_prmtop")
    inpcrd_path = amber_outputs.get("system_inpcrd")
    output_pdb = openmm_outputs.get("minimized_pdb")
    state_xml = openmm_outputs.get("state_xml")
    metrics_json = analysis_outputs.get("metrics_json")

    if not prmtop_path:
        raise ValueError("manifest 中缺少 outputs.amber.system_prmtop")
    if not inpcrd_path:
        raise ValueError("manifest 中缺少 outputs.amber.system_inpcrd")
    if not output_pdb:
        raise ValueError("manifest 中缺少 outputs.openmm.minimized_pdb")
    if not state_xml:
        raise ValueError("manifest 中缺少 outputs.openmm.state_xml")
    if not metrics_json:
        raise ValueError("manifest 中缺少 outputs.analysis.metrics_json")

    summary = run_openmm_minimization(
        output_pdb=output_pdb,
        prmtop_path=prmtop_path,
        inpcrd_path=inpcrd_path,
        state_xml_path=state_xml,
        summary_json_path=metrics_json,
        max_iterations=int(openmm_cfg.get("minimize_max_iterations", max_iterations)),
        nonbonded_cutoff_nm=float(openmm_cfg.get("nonbonded_cutoff_nm", 1.0)),
        constraints=str(openmm_cfg.get("constraints", "HBonds")),
        temperature_K=float(openmm_cfg.get("temperature_K", 300.0)),
        friction_per_ps=float(openmm_cfg.get("friction_per_ps", 1.0)),
        timestep_fs=float(openmm_cfg.get("timestep_fs", 2.0)),
        platform_preference=openmm_cfg.get("platform_preference", ["CUDA", "OpenCL", "CPU"]),
        verbose=True,
    )

    packmol_pdb = ((item.get("system_inputs") or {}).get("packmol_pdb") or "")
    result_check = {}
    stability_index = None
    if packmol_pdb and os.path.exists(packmol_pdb) and output_pdb and os.path.exists(output_pdb):
        try:
            result_check = _compute_result_check(packmol_pdb, output_pdb)
            stability_index = _safe_float(result_check.get("stability_index"), None)
        except Exception as e:
            result_check = {"error": f"result_check_failed: {e}"}

    md_metrics = {
        "mode": "openmm_amber_minimize",
        "openmm_min_pdb": output_pdb,
        "stability_index": stability_index,
        "energy_relaxation": {
            "avg_energy_after_per_atom_kj_per_mol": _safe_float(
                summary.get("avg_energy_after_per_atom_kj_per_mol"), None
            ),
        },
        "geometry_check": {
            "all_atom_rmsd_A": None if not result_check else result_check.get("all_atom_rmsd_A"),
        },
        "packaging_check": {
            "drug_localization_mean_before_A": None if not result_check else result_check.get("drug_radius_mean_before_A"),
            "drug_localization_mean_after_A": None if not result_check else result_check.get("drug_radius_mean_after_A"),
            "drug_localization_shift_A": None if not result_check else result_check.get("drug_radius_mean_shift_A"),
        },
        "details": {
            "platform": summary.get("platform"),
            "periodic": summary.get("periodic"),
            "nonbonded_method": summary.get("nonbonded_method"),
            "output_pdb": summary.get("output_pdb"),
            "state_xml_path": summary.get("state_xml_path"),
            "summary_json_path": summary.get("summary_json_path"),
        },
    }

    _write_json(metrics_json, md_metrics)
    return md_metrics


def run_delivery_pipeline(admet_csv_path: str, config: dict | None = None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg["weights"] = {**DEFAULT_CONFIG["weights"], **(config.get("weights", {}) if config else {})}

    cfg["work_root"] = _resolve_under_eng(cfg.get("work_root", DEFAULT_CONFIG["work_root"]))
    cfg["component_dir"] = _resolve_under_eng(cfg.get("component_dir", DEFAULT_CONFIG["component_dir"]))
    os.makedirs(cfg["work_root"], exist_ok=True)

    admet_csv_path = str(Path(admet_csv_path).resolve())
    admet_csv = Path(admet_csv_path)

    if not admet_csv.exists():
        raise FileNotFoundError(f"ADMET CSV 不存在: {admet_csv}")

    if admet_csv.stat().st_size == 0:
        raise ValueError(f"ADMET CSV 是空文件: {admet_csv}")

    candidates = load_admet_candidates(admet_csv_path, max_candidates=int(cfg["max_candidates"]))
    if not candidates:
        return []

    designs = []
    for candidate in candidates:
        local_designs = enumerate_designs(
            candidate=candidate,
            out_root=cfg["work_root"],
            n_each=int(cfg["structures_per_candidate"]),
        )

        for design in local_designs:
            design = run_packmol(
                design=design,
                packmol_bin=cfg["packmol_bin"],
                component_dir=cfg["component_dir"],
            )

            if not design.get("packmol_ok", False):
                dbg = design.get("packmol_debug", {}) or {}
                design["md_metrics"] = {
                    "mode": "skipped",
                    "openmm_min_pdb": None,
                    "stability_index": None,
                    "details": {
                        "reason": "packmol_failed",
                        "error": dbg.get("error"),
                        "which_packmol_bin": dbg.get("which_packmol_bin"),
                        "resolved_packmol": dbg.get("resolved_packmol"),
                        "component_dir": dbg.get("component_dir"),
                        "spec_path": dbg.get("spec_path"),
                        "spec_exists": dbg.get("spec_exists"),
                        "log_tail": dbg.get("log_tail"),
                    },
                }
                designs.append(design)
                continue

            try:
                run_dir = Path(design.get("work_dir") or design.get("out_dir") or Path(design["packmol_pdb"]).parent)
                result_once_path = run_dir / "result_once_for_manifest.json"
                manifest_path = run_dir / "agent_manifest.json"

                temp_result = _build_temp_result_for_manifest(design)
                _write_json(result_once_path, temp_result)

                build_agent_manifest_from_result_once(
                    result_once_path=Path(result_once_path),
                    template_dir=Path(cfg["component_dir"]),
                    out_path=Path(manifest_path),
                )
                design["manifest_path"] = str(manifest_path)

                build_amber_system_from_manifest(
                    manifest_path=str(manifest_path),
                    candidate_index=0,
                    overwrite=True,
                )

                if bool(cfg.get("enable_openmm", True)):
                    design["md_metrics"] = _run_real_md_from_manifest(
                        manifest_path=str(manifest_path),
                        max_iterations=int(cfg["openmm_max_iterations"]),
                    )
                else:
                    design["md_metrics"] = {
                        "mode": "skipped",
                        "openmm_min_pdb": None,
                        "stability_index": None,
                        "details": {"reason": "openmm_disabled"},
                    }

            except Exception as e:
                design["md_metrics"] = {
                    "mode": "failed",
                    "openmm_min_pdb": None,
                    "stability_index": None,
                    "details": {"error": str(e)},
                }

            designs.append(design)

    scored = score_designs(designs, weights=cfg["weights"])
    return _build_outputs_per_molecule(scored)