import json
import os


def _ensure_dir(p: str):
    d = os.path.dirname(os.path.abspath(p))
    if d:
        os.makedirs(d, exist_ok=True)


def _write_metrics(path: str, metrics: dict):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _packaging_to_stability(packaging_metrics: dict | None):
    if not isinstance(packaging_metrics, dict):
        return None
    try:
        delta_min = packaging_metrics.get("delta_min_A", None)
        nn_mean = packaging_metrics.get("drug_nn_dist_mean_A", None)
        radius_mean = packaging_metrics.get("drug_radius_mean_A", None)

        score = 0.5
        if delta_min is not None:
            score += max(-0.4, min(0.3, float(delta_min) / 50.0))
        if nn_mean is not None:
            score += max(-0.2, min(0.2, (float(nn_mean) - 3.0) / 20.0))
        if radius_mean is not None:
            score += max(-0.1, min(0.1, 1.0 - abs(float(radius_mean) - 80.0) / 120.0))
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return None


def run_md_validation(
    design: dict,
    enable_openmm: bool = True,
    analysis_json_name: str = "md_metrics.json",
    openmm_repulsion_k: float = 10.0,
    openmm_sigma_nm: float = 0.25,
    openmm_radial_k: float = 0.0,
    openmm_max_iterations: int = 2000,
    openmm_tolerance_kj_mol_nm: float = 10.0,
):
    packmol_pdb = design.get("packmol_pdb")
    work_dir = design.get("work_dir") or design.get("out_dir") or design.get("output_dir") or design.get("workdir")
    if not work_dir and packmol_pdb:
        work_dir = os.path.dirname(os.path.abspath(packmol_pdb))
    if not work_dir:
        work_dir = "."

    md_metrics_path = os.path.join(work_dir, analysis_json_name)
    md_metrics = {
        "mode": "openmm",
        "openmm_min_pdb": None,
        "stability_index": None,
        "details": {},
    }

    if not packmol_pdb or (not os.path.exists(packmol_pdb)):
        md_metrics["mode"] = "failed"
        md_metrics["details"]["error"] = f"packmol_pdb not found: {packmol_pdb}"
        _write_metrics(md_metrics_path, md_metrics)
        design["md_metrics"] = md_metrics
        return design

    if not enable_openmm:
        md_metrics["mode"] = "skipped"
        md_metrics["details"]["reason"] = "enable_openmm=False"
        _write_metrics(md_metrics_path, md_metrics)
        design["md_metrics"] = md_metrics
        return design

    try:
        from .openmm_minimizer import run_openmm_minimization, compute_packaging_metrics
    except Exception as e:
        md_metrics["mode"] = "unavailable"
        md_metrics["details"]["error"] = f"OpenMM import failed: {repr(e)}"
        _write_metrics(md_metrics_path, md_metrics)
        design["md_metrics"] = md_metrics
        return design

    openmm_min_pdb = os.path.join(work_dir, "system_openmm_min.pdb")

    try:
        outpdb = run_openmm_minimization(
            input_pdb=packmol_pdb,
            output_pdb=openmm_min_pdb,
            max_iterations=int(openmm_max_iterations),
            repulsion_k=float(openmm_repulsion_k),
            sigma_nm=float(openmm_sigma_nm),
            cutoff_nm=1.0,
            platform_name="CPU",
            stages=3,
        )
        md_metrics["openmm_min_pdb"] = outpdb

        packaging_metrics = None
        try:
            packaging_metrics = compute_packaging_metrics(
                pdb_path=openmm_min_pdb,
                core_radius_A=125.0,
                drug_chain_id="D",
                atoms_per_drug=57,
            )
            md_metrics["packaging_metrics"] = packaging_metrics
        except Exception as e:
            md_metrics.setdefault("details", {})["packaging_metrics_error"] = repr(e)

        md_metrics["stability_index"] = _packaging_to_stability(packaging_metrics)
        if md_metrics["stability_index"] is None:
            md_metrics["stability_index"] = 1.0
        md_metrics["mode"] = "openmm_repulsion"

    except Exception as e:
        md_metrics["mode"] = "failed"
        md_metrics["details"]["error"] = repr(e)

    _write_metrics(md_metrics_path, md_metrics)
    design["md_metrics"] = md_metrics
    return design
