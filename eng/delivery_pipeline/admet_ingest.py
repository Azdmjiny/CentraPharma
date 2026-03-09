import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED


def _pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _calc_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "tPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "RotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "QED": float(QED.qed(mol)),
    }


def load_admet_candidates(csv_path: str, max_candidates: int = 12):
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    smiles_col = _pick_col(df, ["smiles", "SMILES", "canonical_smiles"])
    if smiles_col is None:
        raise ValueError("CSV中未找到SMILES列（支持 smiles/SMILES/canonical_smiles）")

    bbb_col = _pick_col(df, ["BBB", "bbb", "BBB_Martins"])
    herg_col = _pick_col(df, ["hERG", "herg"])
    ames_col = _pick_col(df, ["AMES", "ames"])
    caco2_col = _pick_col(df, ["Caco2", "caco2"])
    qed_col = _pick_col(df, ["QED", "qed"])

    # ——新增：可选列（有就读，没有就算了）——
    pka_col = _pick_col(df, ["pKa", "pka", "pKa_base", "pKa_basic"])
    logd_col = _pick_col(df, ["logD", "logD74", "logD_7.4", "logD7.4", "logD@7.4"])
    tm_col = _pick_col(df, ["Tm", "tm", "melting_point", "meltingPoint", "mp"])
    sol_col = _pick_col(df, ["Solubility", "solubility", "S", "aqueous_solubility"])

    out = []
    for i, row in df.iterrows():
        if len(out) >= max_candidates:
            break

        smiles = str(row.get(smiles_col, "")).strip()
        if not smiles:
            continue

        desc = _calc_descriptors(smiles)
        if desc is None:
            continue

        admet = {
            "BBB": _safe_float(row.get(bbb_col), 0.0) if bbb_col else 0.0,
            "hERG": _safe_float(row.get(herg_col), 0.0) if herg_col else 0.0,
            "AMES": _safe_float(row.get(ames_col), 0.0) if ames_col else 0.0,
            "Caco2": _safe_float(row.get(caco2_col), 0.0) if caco2_col else 0.0,
            "QED": _safe_float(row.get(qed_col), desc.get("QED", 0.0)) if qed_col else desc.get("QED", 0.0),

            # ——新增：可选字段（缺失时用 None，不参与决策）——
            "pKa": _safe_float(row.get(pka_col), None) if pka_col else None,
            "logD74": _safe_float(row.get(logd_col), None) if logd_col else None,
            "Tm": _safe_float(row.get(tm_col), None) if tm_col else None,
            "Solubility": _safe_float(row.get(sol_col), None) if sol_col else None,
        }

        out.append({
            "candidate_id": f"CAND_{i+1:04d}",
            "smiles": smiles,
            "descriptors": desc,
            "admet": admet,
            "raw": {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        })

    return out