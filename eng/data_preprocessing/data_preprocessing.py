import argparse
import contextlib
import io
import math
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem
from pkasolver.query import calculate_microstate_pka_values
import pkasolver

def is_missing(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"", "nan", "none", "null", "na", "n/a"}:
            return True
    return False


def pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def safe_float(x):
    try:
        if is_missing(x):
            return None
        return float(x)
    except Exception:
        return None


def resolve_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent

    # 你现在脚本在 eng/delivery_pipeline/ 下，
    # 真正想要的是 eng/output/
    candidates = [
        script_dir.parent / "output",          # eng/output
        script_dir.parent.parent / "output",   # 再往上一层兜底
    ]

    for p in candidates:
        if p.is_dir():
            return p

    raise FileNotFoundError(
        "没找到 output 文件夹。已检查：\n"
        + "\n".join(str(p) for p in candidates)
    )


def resolve_input_csv(output_dir: Path, explicit_input: str | None) -> Path:
    if explicit_input:
        p = Path(explicit_input).resolve()
        if not p.exists():
            raise FileNotFoundError(f"输入 CSV 不存在: {p}")
        return p

    # 优先原始 material_classified_all.csv；否则找最新的非 enriched csv
    preferred = output_dir / "material_classified_all.csv"
    if preferred.exists():
        return preferred

    csvs = [
        p for p in output_dir.glob("*.csv")
        if not p.name.endswith("_enriched.csv")
    ]
    if not csvs:
        raise FileNotFoundError(f"在 {output_dir} 下没找到可用 CSV")
    return max(csvs, key=lambda p: p.stat().st_mtime)


def resolve_output_csv(output_dir: Path, input_csv: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).resolve()
    return output_dir / f"{input_csv.stem}_enriched.csv"


def extract_pka_values(smiles: str):
    """
    返回：
        pka_values: 所有有效 pKa 的列表
        pka_mean: 所有有效 pKa 的平均值（只是汇总展示用）
        pka_near_7: 最接近 pH 7.0 的 pKa
        pka_near_74: 最接近 pH 7.4 的 pKa
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], None, None, None

    fake_out = io.StringIO()
    fake_err = io.StringIO()

    try:
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_err):
            results = calculate_microstate_pka_values(mol)
    except Exception as e:
        msg = str(e)
        if "Could not identify any ionizable group" in msg:
            return [], None, None, None
        return [], None, None, None

    if not results:
        return [], None, None, None

    values = []
    for state in results:
        pka = getattr(state, "pka", None)
        if pka is None:
            continue
        try:
            pka = float(pka)
        except Exception:
            continue
        if math.isnan(pka):
            continue
        values.append(pka)

    if not values:
        return [], None, None, None

    pka_mean = sum(values) / len(values)
    pka_near_7 = min(values, key=lambda x: abs(x - 7.0))
    pka_near_74 = min(values, key=lambda x: abs(x - 7.4))

    return values, pka_mean, pka_near_7, pka_near_74

def calc_logd_from_logp_and_pka_nearest(logp, pka_nearest, ph):
    """
    工程近似：
    只使用“最接近当前 pH 的 pKa”来估计 logD。

    近似式：
        logD = logP - log10(1 + 10^|pKa - pH|)

    解释：
    - 当 pKa 越接近当前 pH，离子化越显著，logD 相对 logP 降低越明显
    - 当 pKa 离当前 pH 很远，修正项变小/或表现为更弱影响
    - 这是工程近似，不是严格微观状态分布模型
    """
    if is_missing(logp):
        return None
    if is_missing(pka_nearest):
        return None

    try:
        logp = float(logp)
        pka_nearest = float(pka_nearest)
    except Exception:
        return None

    return logp - math.log10(1.0 + 10.0 ** abs(pka_nearest - ph))

def build_clean_row(row, smiles_col, pka_mean, pka_near_7, pka_near_74, logd7, logd74):
    """
    只保留真正需要的输入列 + 新生成列。
    明确丢弃旧分类结果列，避免 enriched 表把历史 PLGA/decision_* 带进去。
    """
    keep_candidates = [
        "mol_id",
        smiles_col,
        "candidate_id",
        "MW",
        "logP",
        "tPSA",
        "HBA",
        "HBD",
        "RotB",
        "QED_rdkit",
        "BBB",
        "BBB_Martins",
        "hERG",
        "AMES",
        "Caco2",
        "QED_ingest",
        "cargo_type",
        "molecule_type",
        "acid_base",
        "is_nucleic_acid",
    ]

    out = {}
    for col in keep_candidates:
        if col in row.index:
            out[col] = row.get(col)

    # 统一 smiles 列名
    if smiles_col in out and smiles_col != "smiles":
        out["smiles"] = out.pop(smiles_col)

    # 兼容上游 ADMETModel 列名：很多流程输出 BBB_Martins 而不是 BBB。
    # 下游 delivery_pipeline 优先读取 BBB，因此这里补齐同义字段，避免 BBB 变成默认 0.0。
    if is_missing(out.get("BBB")) and not is_missing(out.get("BBB_Martins")):
        out["BBB"] = out.get("BBB_Martins")

    out["pKa"] = pka_mean
    out["pKa_near_7"] = pka_near_7
    out["pKa_near_74"] = pka_near_74
    out["logD7"] = logd7
    out["logD74"] = logd74

    return out

def main():
    parser = argparse.ArgumentParser(
        description="自动读取 output 下的 CSV，计算 pKa，并整理 logD7/logD74。算不出来 pKa 的分子直接删除。"
    )
    parser.add_argument("--input-csv", default=None, help="可选：手动指定输入 CSV")
    parser.add_argument("--output-csv", default=None, help="可选：手动指定输出 CSV")
    args = parser.parse_args()

    # 让 pkasolver 内部临时写文件时一定在可写目录
    safe_workdir = Path(tempfile.gettempdir()) / "pkasolver_safe_workdir"
    safe_workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(safe_workdir)

    pkg_site = str(Path(pkasolver.__file__).resolve().parent.parent)
    os.environ["PYTHONPATH"] = pkg_site + os.pathsep + os.environ.get("PYTHONPATH", "")

    # 关键修复：
    # 如果显式传了输入输出，就不要再强依赖默认 output_dir
    if args.input_csv and args.output_csv:
        input_csv = Path(args.input_csv).resolve()
        output_csv = Path(args.output_csv).resolve()
        if not input_csv.exists():
            raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")
    else:
        output_dir = resolve_output_dir()
        input_csv = resolve_input_csv(output_dir, args.input_csv)
        output_csv = resolve_output_csv(output_dir, input_csv, args.output_csv)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"输入 CSV 为空: {input_csv}")

    smiles_col = pick_col(df, ["smiles", "SMILES", "canonical_smiles"])
    if smiles_col is None:
        raise ValueError("CSV 中未找到 smiles 列")

    total = len(df)
    kept_rows = []

    for _, row in df.iterrows():
        smiles = str(row.get(smiles_col, "")).strip()
        if not smiles:
            continue

        pka_values, pka_mean, pka_near_7, pka_near_74 = extract_pka_values(smiles)

        # 算不出来 pKa 的，直接丢弃
        if len(pka_values) == 0:
            continue

        logp_val = row.get("logP")

        # 不再复用旧的 logD7/logD74 列，统一重新按当前逻辑计算
        logd7 = calc_logd_from_logp_and_pka_nearest(
            logp=logp_val,
            pka_nearest=pka_near_7,
            ph=7.0,
        )

        logd74 = calc_logd_from_logp_and_pka_nearest(
            logp=logp_val,
            pka_nearest=pka_near_74,
            ph=7.4,
        )

        clean_row = build_clean_row(
            row=row,
            smiles_col=smiles_col,
            pka_mean=pka_mean,
            pka_near_7=pka_near_7,
            pka_near_74=pka_near_74,
            logd7=logd7,
            logd74=logd74,
        )
        kept_rows.append(clean_row)

    out_df = pd.DataFrame(kept_rows)

    if not kept_rows:
        raise RuntimeError(
            f"pKa 预处理后 0 条候选被保留。"
            f" total={total}，请检查 pkasolver/dimorphite 是否正常。"
        )

    preferred_order = [
        "mol_id",
        "smiles",
        "candidate_id",
        "MW",
        "logP",
        "tPSA",
        "HBA",
        "HBD",
        "RotB",
        "QED_rdkit",
        "BBB",
        "BBB_Martins",
        "hERG",
        "AMES",
        "Caco2",
        "QED_ingest",
        "cargo_type",
        "molecule_type",
        "acid_base",
        "is_nucleic_acid",
        "pKa",
        "pKa_near_7",
        "pKa_near_74",
        "logD7",
        "logD74",
    ]
    existing_cols = [c for c in preferred_order if c in out_df.columns]
    other_cols = [c for c in out_df.columns if c not in existing_cols]
    out_df = out_df[existing_cols + other_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tmp_output_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    out_df.to_csv(tmp_output_csv, index=False, encoding="utf-8-sig")

    # 原子替换，避免下游读到半成品/空文件
    os.replace(tmp_output_csv, output_csv)

    success = len(out_df)
    deleted = total - success

    print(f"[INFO] input_csv   = {input_csv}")
    print(f"[INFO] output_csv  = {output_csv}")
    print(f"[INFO] total       = {total}")
    print(f"[INFO] kept        = {success}")
    print(f"[INFO] deleted     = {deleted}")
    print("[DONE]")

if __name__ == "__main__":
    main()
