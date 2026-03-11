import json
import os

from .openmm_minimizer import run_openmm_minimization


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_json(path: str, obj: dict):
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def run_md_validation_from_manifest(
    manifest_path: str,
    candidate_index: int = 0,
    max_iterations: int = 5000,
):
    """
    新版真实最小化入口：
    - 读取 agent_manifest.json
    - 找到 amber 输出的 prmtop / inpcrd
    - 调用真实 OpenMM 最小化
    - 写回 metrics_json

    返回结构保持尽量兼容旧 pipeline：
    {
        "mode": "openmm_amber_minimize",
        "openmm_min_pdb": "...",
        "stability_index": None,
        "details": {...}
    }
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    manifest = _load_json(manifest_path)
    candidates = manifest.get("candidates", [])

    if not isinstance(candidates, list) or len(candidates) == 0:
        raise ValueError("manifest 中没有 candidates")

    if candidate_index < 0 or candidate_index >= len(candidates):
        raise IndexError(
            f"candidate_index 越界: {candidate_index}, candidates 数量={len(candidates)}"
        )

    item = candidates[candidate_index]

    prmtop_path = _safe_get(item, "outputs", "amber", "system_prmtop")
    inpcrd_path = _safe_get(item, "outputs", "amber", "system_inpcrd")
    output_pdb = _safe_get(item, "outputs", "openmm", "minimized_pdb")
    state_xml = _safe_get(item, "outputs", "openmm", "state_xml")
    metrics_json = _safe_get(item, "outputs", "analysis", "metrics_json")

    if not prmtop_path:
        raise ValueError("manifest 缺少 outputs.amber.system_prmtop")
    if not inpcrd_path:
        raise ValueError("manifest 缺少 outputs.amber.system_inpcrd")
    if not output_pdb:
        raise ValueError("manifest 缺少 outputs.openmm.minimized_pdb")
    if not state_xml:
        raise ValueError("manifest 缺少 outputs.openmm.state_xml")
    if not metrics_json:
        raise ValueError("manifest 缺少 outputs.analysis.metrics_json")

    try:
        result = run_openmm_minimization(
            prmtop_path=prmtop_path,
            inpcrd_path=inpcrd_path,
            output_pdb=output_pdb,
            state_xml_path=state_xml,
            summary_json_path=metrics_json,
            max_iterations=max_iterations,
        )

        md_metrics = {
            "mode": "openmm_amber_minimize",
            "openmm_min_pdb": output_pdb,
            "stability_index": None,
            "details": result,
        }

        # 为了让 pipeline 读取时统一，这里把兼容结构覆盖写回 metrics_json
        _write_json(metrics_json, md_metrics)
        return md_metrics

    except Exception as e:
        md_metrics = {
            "mode": "failed",
            "openmm_min_pdb": None,
            "stability_index": None,
            "details": {
                "error": str(e),
                "manifest_path": os.path.abspath(manifest_path),
                "prmtop_path": os.path.abspath(prmtop_path) if prmtop_path else None,
                "inpcrd_path": os.path.abspath(inpcrd_path) if inpcrd_path else None,
            },
        }
        _write_json(metrics_json, md_metrics)
        return md_metrics


def run_md_validation(
    design: dict,
    analysis_json_name: str = "md_metrics.json",
    openmm_max_iterations: int = 5000,
    **kwargs,
):
    """
    兼容旧 pipeline 的包装器。
    现在要求 design 里提供 manifest_path 或 agent_manifest 路径。

    design 至少应包含：
    {
        "manifest_path": ".../agent_manifest.json"
    }

    如果你的 pipeline 还没改完，只调用旧的 packmol_pdb，
    那这里会明确报错，而不是偷偷走旧软排斥逻辑。
    """
    manifest_path = (
        design.get("manifest_path")
        or design.get("agent_manifest")
        or design.get("agent_manifest_path")
    )

    if not manifest_path:
        work_dir = (
            design.get("work_dir")
            or design.get("out_dir")
            or design.get("output_dir")
            or "."
        )
        metrics_json = os.path.join(work_dir, analysis_json_name)
        md_metrics = {
            "mode": "failed",
            "openmm_min_pdb": None,
            "stability_index": None,
            "details": {
                "error": "design 中缺少 manifest_path / agent_manifest_path，无法进行真实最小化"
            },
        }
        _write_json(metrics_json, md_metrics)
        return md_metrics

    return run_md_validation_from_manifest(
        manifest_path=manifest_path,
        candidate_index=int(design.get("candidate_index", 0) or 0),
        max_iterations=int(openmm_max_iterations),
    )