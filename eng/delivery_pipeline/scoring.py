def _clamp01(x: float):
    return max(0.0, min(1.0, float(x)))


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _score_material(desc: dict, admet: dict):
    logp = float(desc.get("logP", 0.0))
    tpsa = float(desc.get("tPSA", 200.0))
    mw = float(desc.get("MW", 1000.0))
    bbb = float(admet.get("BBB", 0.0))

    s_logp = _clamp01(1.0 - abs(logp - 3.0) / 2.5)
    s_tpsa = _clamp01(1.0 - min(tpsa / 90.0, 1.0))
    s_mw = _clamp01(1.0 - abs(mw - 350.0) / 250.0)
    s_bbb = _clamp01(min(bbb / 0.70, 1.0))
    return 0.35 * s_logp + 0.25 * s_tpsa + 0.20 * s_mw + 0.20 * s_bbb


def _score_structure(design: dict):
    s_size = _clamp01(1.0 - abs(float(design["size_nm"]) - 100.0) / 40.0)
    s_zeta = _clamp01(1.0 - abs(float(design["zeta_mv"])) / 20.0)
    s_load = _clamp01(1.0 - abs(float(design["drug_loading"]) - 0.12) / 0.08)
    s_pack = 1.0 if design.get("packmol_ok", False) else 0.25
    return 0.35 * s_size + 0.25 * s_zeta + 0.20 * s_load + 0.20 * s_pack


def _score_md(md: dict):
    stability = _safe_float(md.get("stability_index"), None)
    if stability is not None:
        return _clamp01(stability)

    packaging = md.get("packaging_metrics", {}) or {}
    delta_min = _safe_float(packaging.get("delta_min_A"), None)
    nn_mean = _safe_float(packaging.get("drug_nn_dist_mean_A"), None)
    radius_mean = _safe_float(packaging.get("drug_radius_mean_A"), None)

    score = 0.5
    used = False
    if delta_min is not None:
        score += max(-0.35, min(0.30, delta_min / 50.0))
        used = True
    if nn_mean is not None:
        score += max(-0.20, min(0.20, (nn_mean - 3.0) / 20.0))
        used = True
    if radius_mean is not None:
        score += max(-0.10, min(0.10, 1.0 - abs(radius_mean - 80.0) / 120.0))
        used = True

    if used:
        return _clamp01(score)

    mode = str(md.get("mode", "") or "").lower()
    if mode in {"openmm_repulsion", "real"}:
        return 0.75
    if mode in {"skipped", "unavailable"}:
        return 0.50
    if mode == "failed":
        return 0.20
    return 0.50


def score_designs(designs: list, weights: dict | None = None):
    if weights is None:
        weights = {"material": 0.30, "structure": 0.30, "md": 0.30, "qed": 0.10}

    scored = []
    for d in designs:
        s_mat = _score_material(d["descriptors"], d["admet"])
        s_struct = _score_structure(d)
        s_md = _score_md(d.get("md_metrics", {}))
        s_qed = _clamp01(d["admet"].get("QED", d["descriptors"].get("QED", 0.0)))

        total = (
            weights["material"] * s_mat
            + weights["structure"] * s_struct
            + weights["md"] * s_md
            + weights["qed"] * s_qed
        )

        scored.append({
            "design": d,
            "score_total": float(total),
            "score_breakdown": {
                "S_material": float(s_mat),
                "S_structure": float(s_struct),
                "S_md": float(s_md),
                "S_qed": float(s_qed),
            },
        })

    scored.sort(key=lambda x: x["score_total"], reverse=True)
    return scored
