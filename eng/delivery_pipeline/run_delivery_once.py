import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 优先尊重外部环境变量；否则让 packmol 走 PATH
os.environ.setdefault("PACKMOL_BIN", "packmol")


def _resolve_templates() -> str:
    candidates = [
        DELIVERY_ROOT / "template",
        PROJECT_ROOT / "LNP" / "templates",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    return str(candidates[0])


def _find_latest_csv(search_root: Path) -> Path:
    filtered_csvs = list(search_root.rglob("*_filtered.csv"))
    if filtered_csvs:
        return max(filtered_csvs, key=lambda p: p.stat().st_mtime)

    all_csvs = list(search_root.rglob("*.csv"))
    if all_csvs:
        return max(all_csvs, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"在 {search_root} 下没找到任何 csv")


def _resolve_entry_function():
    mod = importlib.import_module("delivery_pipeline.pipeline")
    for name in ("run_delivery_pipeline", "run_pipeline", "main"):
        if hasattr(mod, name) and callable(getattr(mod, name)):
            return getattr(mod, name)
    raise RuntimeError("找不到入口函数 run_delivery_pipeline/run_pipeline/main")


def _build_kwargs(fn, single_csv: Path, work_root: Path) -> dict:
    sig = inspect.signature(fn)
    kwargs = {}

    for p in sig.parameters.values():
        n = p.name.lower()

        if n in ("admet_csv", "admet_csv_path", "csv_path", "input_csv", "csv_file", "input_file"):
            kwargs[p.name] = str(single_csv)

        elif n == "config":
            kwargs[p.name] = {
                "max_candidates": 1,
                "structures_per_candidate": 1,
                "work_root": str(work_root),
                "packmol_bin": os.environ.get("PACKMOL_BIN", "packmol"),
                "component_dir": _resolve_templates(),
                "enable_openmm": os.environ.get("ENABLE_OPENMM", "1") not in {"0", "false", "False"},
            }

        elif n in ("work_root", "output_dir", "out_dir"):
            kwargs[p.name] = str(work_root)

        elif n in ("packmol_bin", "packmol_path"):
            kwargs[p.name] = os.environ.get("PACKMOL_BIN", "packmol")

        elif p.default is inspect._empty:
            raise RuntimeError(f"入口函数有无法自动注入的必填参数: {p.name}")

    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Run one-shot drug delivery pipeline on a specific ADMET CSV.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="要处理的 ADMET 筛选结果 CSV 路径",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="最终结果 JSON 输出路径",
    )
    parser.add_argument(
        "--work-root",
        type=str,
        default=None,
        help="中间产物和结构文件输出目录；默认取 output-json 的父目录，或环境变量 DELIVERY_ONCE_OUT",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="未显式提供 --csv-path 时，用于自动搜索 CSV 的根目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 解析输入 CSV
    if args.csv_path:
        csv_path = Path(args.csv_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"--csv-path 指定的文件不存在: {csv_path}")
    else:
        search_root = Path(
            args.input_root
            or os.environ.get("DELIVERY_INPUT_ROOT", str(PROJECT_ROOT / "output"))
        ).resolve()
        csv_path = _find_latest_csv(search_root)

    # 2. 解析输出 JSON 路径
    if args.output_json:
        output_json = Path(args.output_json).resolve()
    else:
        default_out_dir = Path(
            args.work_root
            or os.environ.get("DELIVERY_ONCE_OUT", str(DELIVERY_ROOT / "output" / "delivery_runs_once"))
        ).resolve()
        output_json = default_out_dir / "result_once.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)

    # 3. 解析 work_root
    if args.work_root:
        work_root = Path(args.work_root).resolve()
    else:
        # 最稳：默认和 output_json 放同一个目录
        work_root = output_json.parent.resolve()

    work_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] csv_path   =", csv_path)
    print("[INFO] work_root  =", work_root)
    print("[INFO] output_json=", output_json)

    # 4. 只取一条分子，符合 once 模式
    single_csv = work_root / "single_molecule.csv"
    df = pd.read_csv(csv_path).head(1).copy()
    df.to_csv(single_csv, index=False, encoding="utf-8-sig")

    # 5. 调 pipeline 入口
    fn = _resolve_entry_function()
    kwargs = _build_kwargs(fn, single_csv=single_csv, work_root=work_root)

    print("[INFO] pipeline entry =", fn.__module__ + "." + fn.__name__)
    print("[INFO] injected kwargs keys =", list(kwargs.keys()))

    res = fn(**kwargs)

    # 6. 写最终 JSON
    output_json.write_text(
        json.dumps(res, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] {output_json}")
    print(json.dumps(res, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()