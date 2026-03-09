import os
from pathlib import Path

from .admet_ingest import load_admet_candidates
from .structure_packmol import enumerate_designs, run_packmol
from .md_validator import run_md_validation
from .scoring import score_designs

DEFAULT_CONFIG = {
    "max_candidates": 8,
    "structures_per_candidate": 4,
    # 一律按 eng 根目录下的相对路径解释
    "work_root": "output/delivery_runs",
    # packmol 可执行名；若已加入 PATH，直接保留 'packmol' 即可
    "packmol_bin": "packmol",
    # 现在默认优先读 delivery_pipeline/template
    "component_dir": "delivery_pipeline/template",
    # OpenMM 开关：没装时 md_validator 也会自动降级，不会整条链路直接 import 崩溃
    "enable_openmm": True,
    "analysis_json_name": "md_metrics.json",
    "openmm_repulsion_k": 10.0,
    "openmm_sigma_nm": 0.25,
    "openmm_radial_k": 1.0,
    "openmm_max_iterations": 2000,
    "openmm_tolerance_kj_mol_nm": 10.0,
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


def _build_single_output(best_item: dict) -> dict:
    design = best_item.get("design", {}) or {}
    desc = design.get("descriptors", {}) or {}
    admet = design.get("admet", {}) or {}
    md_metrics = design.get("md_metrics", {}) or {}
    score_breakdown = best_item.get("score_breakdown", {}) or {}

    strategy = str(design.get("strategy", "") or "").lower()
    material = design.get("material", None)

    if "liposome" in strategy or str(material).upper() == "LIPOSOME":
        delivery_type = "liposome"
    elif str(material).upper() == "LNP":
        delivery_type = "lipid_nanoparticle"
    elif str(material).upper() == "NLC":
        delivery_type = "nanostructured_lipid_carrier"
    else:
        delivery_type = "nanoparticle"

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
        advantages.append(f"已完成结构规整与指标计算({md_mode})，稳定性指标={stability_index:.3f}")
    elif md_mode in {"skipped", "unavailable"}:
        advantages.append("OpenMM 暂未接通，当前仅保留结构构建结果")
    else:
        advantages.append("已完成结构规整与指标计算")

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


def run_delivery_pipeline(admet_csv_path: str, config: dict | None = None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg["weights"] = {**DEFAULT_CONFIG["weights"], **(config.get("weights", {}) if config else {})}

    cfg["work_root"] = _resolve_under_eng(cfg.get("work_root", DEFAULT_CONFIG["work_root"]))
    cfg["component_dir"] = _resolve_under_eng(cfg.get("component_dir", DEFAULT_CONFIG["component_dir"]))
    os.makedirs(cfg["work_root"], exist_ok=True)

    candidates = load_admet_candidates(admet_csv_path, max_candidates=int(cfg["max_candidates"]))
    if not candidates:
        return []

    designs = []
    for candidate in candidates:
        local_designs = enumerate_designs(candidate=candidate, out_root=cfg["work_root"], n_each=int(cfg["structures_per_candidate"]))
        for design in local_designs:
            design = run_packmol(design=design, packmol_bin=cfg["packmol_bin"], component_dir=cfg["component_dir"])
            design = run_md_validation(
                design=design,
                enable_openmm=bool(cfg.get("enable_openmm", True)),
                analysis_json_name=cfg["analysis_json_name"],
                openmm_repulsion_k=cfg["openmm_repulsion_k"],
                openmm_sigma_nm=cfg["openmm_sigma_nm"],
                openmm_radial_k=cfg["openmm_radial_k"],
                openmm_max_iterations=cfg["openmm_max_iterations"],
                openmm_tolerance_kj_mol_nm=cfg["openmm_tolerance_kj_mol_nm"],
            )
            designs.append(design)

    scored = score_designs(designs, weights=cfg["weights"])
    return _build_outputs_per_molecule(scored)
