# tools.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import csv
import subprocess
from pathlib import Path

import pandas as pd
import torch

from core import BASE, suppress_rdkit_warnings

from admet_ai import ADMETModel
import mygene
from bioservices import UniProt
from Bio import SeqIO

from typing import Annotated
from uuid import uuid4

# 初始化全局模型
_global_model = None


def get_admet_model():
    """获取全局ADMET模型实例"""
    global _global_model
    if _global_model is None:
        with suppress_rdkit_warnings():
            _global_model = ADMETModel()
    return _global_model


def admet_filter_tool(input_dir: str) -> dict:
    """执行ADMET筛选的核心工具函数"""
    try:
        if not os.path.isdir(input_dir):
            raise ValueError(f"无效的输入目录: {input_dir}")

        model = get_admet_model()

        out_dir = (BASE / ".." / "output").resolve()
        os.makedirs(out_dir, exist_ok=True)

        valid_dirs = find_csv_directories(input_dir)
        if not valid_dirs:
            raise FileNotFoundError(
                f"未找到任何包含hash_ligand_mapping.csv的目录，请检查:\n"
                f"1. 文件是否实际存在\n"
                f"2. 目录结构是否符合预期\n"
                f"3. 路径: {input_dir}"
            )

        results = []
        output_files = []

        for dirpath in valid_dirs:
            in_csv = os.path.join(dirpath, "hash_ligand_mapping.csv")
            relative_path = os.path.relpath(dirpath, start=input_dir)
            safe_path = re.sub(r"[^\w]", "_", relative_path)
            out_csv = os.path.join(out_dir, f"{safe_path}_filtered.csv")

            try:
                df = pd.read_csv(in_csv, header=None, names=["hash", "smiles"])
                smiles_list = df["smiles"].tolist()

                preds = model.predict(smiles=smiles_list)
                preds_df = pd.DataFrame(preds)
                preds_df["smiles"] = smiles_list

                mask = (
                    (preds_df["BBB_Martins"] >= 0.40) &
                    (preds_df["QED"] >= 0.35) &
                    (preds_df["logP"].between(0.5, 4.0)) &
                    (preds_df["tpsa"] <= 120) &
                    (preds_df["hERG"] <= 0.5) &
                    (preds_df["AMES"] <= 0.5)
                )

                filtered = preds_df[mask]

                if filtered.empty:
                    if os.path.exists(out_csv):
                        os.remove(out_csv)
                    results.append(f"{relative_path}: 筛选后 0 条分子，跳过输出")
                    continue

                tmp_out_csv = out_csv + ".tmp"
                filtered.to_csv(tmp_out_csv, index=False)
                os.replace(tmp_out_csv, out_csv)

                results.append(f"{relative_path}: 筛选完成，共 {len(filtered)} 条分子 -> {out_csv}")
                output_files.append(out_csv)

            except Exception as e:
                results.append(f"{relative_path}: 处理失败 - {e}")

        return {
            "category": "admet_filter",
            "file_count": len(output_files),
            "file_paths": output_files,
            "details": results,
        }

    except Exception as e:
        return {
            "category": "admet_filter",
            "error": str(e),
            "details": f"{str(e.__class__.__name__)}: {str(e)}",
        }


def disease_to_protein_sequences(disease_name: str) -> dict:
    """
    将疾病名称转换为相关的蛋白靶点序列
    """
    try:
        base_dir = (BASE / "..").resolve()
        output_dir = base_dir / "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = output_dir / f"{disease_name}.csv"

        # 1. 查找与疾病相关的靶点ID
        targets = set()
        file_path = (BASE / ".." / "data" / "P1-06-Target_disease.txt").resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件 {file_path}")

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                target_id, data_type = parts[0], parts[1]

                # 直接从 INDICATI 行提取靶点 ID
                if data_type == "INDICATI":
                    # 简单验证 target_id 格式（如 T12345）
                    if re.match(r"^T\d+$", target_id):
                        # 检查疾病名称是否出现在该行
                        if re.search(rf"\b{re.escape(disease_name.lower())}\b", line.lower()):
                            targets.add(target_id)
                    else:
                        print(f"[DEBUG] 跳过无效的 target_id: {target_id}")

        if not targets:
            raise ValueError(f"未找到与疾病 {disease_name} 相关的靶点。")

        # 2. 将靶点ID保存到CSV文件
        with output_file.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Target_ID"])
            for target in sorted(targets):
                writer.writerow([target])

        # 3. 从FASTA文件中查找蛋白序列
        fasta_path = (BASE / ".." / "data" / "P2-06-TTD_sequence_all.txt").resolve()
        if not fasta_path.exists():
            raise FileNotFoundError(f"找不到文件 {fasta_path}")

        sequences = {}
        current_id = None
        current_seq_parts = []

        with fasta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(">"):
                    if current_id and current_seq_parts:
                        sequences[current_id] = "".join(current_seq_parts)
                    header = line[1:].split("\t")[0].strip()
                    current_id = header
                    current_seq_parts = []
                else:
                    if current_id:
                        current_seq_parts.append(line.strip())

            if current_id and current_seq_parts:
                sequences[current_id] = "".join(current_seq_parts)

        found_sequences = {tid: sequences[tid] for tid in targets if tid in sequences}

        # 4. 将找到的序列写入文件
        fasta_out = output_dir / "found_protein_sequences.fasta"
        with fasta_out.open("w", encoding="utf-8") as f:
            for tid, seq in found_sequences.items():
                f.write(f">{tid}\n")
                for i in range(0, len(seq), 80):
                    f.write(f"{seq[i:i+80]}\n")

        return {
            "category": "disease_protein_targets",
            "disease_name": disease_name,
            "targets": list(found_sequences.keys()),
            "file_path": str(fasta_out),
        }

    except Exception as e:
        return {
            "category": "disease_protein_targets",
            "disease_name": disease_name,
            "targets": [],
            "file_path": None,
            "error": str(e),
        }


def gene_target_to_protein_sequence(gene_name: str) -> dict:
    """
    将基因名称转换为蛋白质序列并保存为FASTA文件
    """
    try:
        # 确保输出目录存在
        output_dir = (BASE / ".." / "tmp").resolve()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fasta = str(output_dir / f"{gene_name}.fasta")

        # 1. 用 mygene 查询基因名称对应的 UniProt ID
        mg = mygene.MyGeneInfo()
        res = mg.query(
            gene_name,
            species="human",
            fields="symbol,uniprot.Swiss-Prot",
            size=10,
        )

        # 获取与基因相关的 UniProt Swiss-Prot ID
        uniprot_ids = set()
        for hit in res.get("hits", []):
            swiss = hit.get("uniprot", {}).get("Swiss-Prot")
            if isinstance(swiss, list):
                uniprot_ids.update(swiss)
            elif isinstance(swiss, str):
                uniprot_ids.add(swiss)

        if not uniprot_ids:
            # 如果mygene找不到，尝试将输入作为UniProt ID处理
            uniprot_ids.add(gene_name)

        # 2. 用 bioservices UniProt 接口按 ID 获取蛋白序列
        u = UniProt()
        fastas = []
        for uid in uniprot_ids:
            try:
                fasta_str = u.retrieve(uid, frmt="fasta")
            except Exception as e:
                print(f"检索 {uid} 时出错：{e}")
                continue

            if fasta_str and fasta_str.startswith(">"):
                fastas.append(fasta_str.strip())
            else:
                print(f"UniProt 上未找到 {uid} 的序列或返回格式不对。")

        if not fastas:
            raise ValueError("未获取到任何 FASTA 序列。")

        # 3. 将所有 FASTA 序列写入文件
        with open(out_fasta, "w") as f:
            f.write("\n".join(fastas))

        # 4. 提取第一条序列
        protein_seq = ""
        for record in SeqIO.parse(out_fasta, "fasta"):
            protein_seq = str(record.seq)
            break

        return {
            "category": "protein_target_seq",
            "gene_target": gene_name,
            "content": protein_seq,
            "file_path": out_fasta,
        }
    except Exception as e:
        return {
            "category": "protein_target_seq",
            "gene_target": gene_name,
            "content": "",
            "file_path": None,
            "error": str(e),
        }


def generate_ligands(protein_sequence: str, target_id: str = None, num_molecules: int = 20) -> str:
    """
    调用 DrugGPT 生成配体，返回 JSON 格式字符串
    """
    try:
        # 修改输出路径为当前目录 output/target_id
        output_dir = (BASE / ".." / "output").resolve()
        if target_id:
            output_dir = output_dir / target_id  # 使用靶点ID作为子目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 将CUDA设备设置为第二张显卡（索引从0开始）
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # 设置输入和输出文件路径
        input_fasta = output_dir / "input.fasta"
        output_dir_min = output_dir / "seq_min"

        # 将蛋白质序列写入 FASTA 文件
        with open(input_fasta, "w") as f:
            f.write(f">protein\n{protein_sequence}")

        # 定义 DrugGPT 命令
        DRUGGPT_SCRIPT = (BASE / ".." / "druggpt" / "drug_generator.py").resolve()

        if not DRUGGPT_SCRIPT.exists():
            raise FileNotFoundError(f"DrugGPT script not found: {DRUGGPT_SCRIPT}")

        cmd = [
            sys.executable,
            str(DRUGGPT_SCRIPT),
            "-p",
            protein_sequence,
            "-n",
            str(num_molecules),
            "-o",
            str(output_dir_min),
        ]

        print("==== BRIDGE DEBUG ====")
        print("PYEXE:", sys.executable)
        print("torch:", torch.__version__)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("is_available:", torch.cuda.is_available())
        print("device_count:", torch.cuda.device_count())
        print("======================")

        # 执行命令
        DRUGGPT_DIR = DRUGGPT_SCRIPT.parent
        result = subprocess.run(cmd, text=True, cwd=str(DRUGGPT_DIR))
        if result.returncode != 0:
            raise RuntimeError(f"DrugGPT 执行失败：{result.stderr}")

        return json.dumps(
            {
                "category": "protein_target_seq",
                "content": f"成功生成 {num_molecules} 个配体",
                "file_path": str(output_dir_min),
                "target_id": target_id,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "category": "protein_target_seq",
                "content": f"生成配体时出错：{str(e)}",
                "file_path": None,
                "target_id": target_id,
            },
            ensure_ascii=False,
        )


def batch_generate_ligands(fasta_path: str, max_targets: int = 2) -> list:
    """
    批量生成配体，最多处理max_targets个靶点
    """
    results = []

    try:
        count = 0
        for record in SeqIO.parse(fasta_path, "fasta"):
            if count >= max_targets:
                break

            target_id = record.id
            protein_seq = str(record.seq)

            print(f"正在处理第 {count + 1} 个靶点: {target_id}")

            # 生成配体
            ligand_result = generate_ligands(protein_seq, target_id)
            ligand_result_dict = json.loads(ligand_result)

            results.append(ligand_result_dict)
            count += 1

        print(f"已完成 {count} 个靶点的配体生成")
        return results
    except Exception as e:
        print(f"批量生成配体时出错: {str(e)}")
        return []


#############################################################
# 新增：递送方案设计 Tool 函数
#############################################################
# def design_drug_delivery_system(admet_csv_path: str) -> dict:
#     """
#     根据ADMET筛选结果的CSV设计药物递送方案（示例实现）
#     """
#     try:
#         df = pd.read_csv(admet_csv_path)
#         # 假设第一行第一列是smiles
#         first_smiles = df['smiles'].iloc[0] if 'smiles' in df.columns else "UNKNOWN"
#         return {
#             "category": "drug_delivery_design",
#             "smiles": first_smiles,
#             "drug_properties": df.iloc[0].to_dict(),
#             "delivery_system": {"type": "nanoparticle", "material": "PEG-PLGA"},
#             "bbb_strategy": {"method": "receptor-mediated transcytosis"},
#             "advantages": ["高生物利用度", "靶向性强", "降低毒副作用"]
#         }
#     except Exception as e:
#         return {
#             "category": "drug_delivery_design",
#             "smiles": "",
#             "error": str(e)
#         }


def _find_conda_env_python(env_name: str) -> str:
    """
    返回指定 conda 环境中的 python 绝对路径。
    优先根据 CONDA_EXE 推导 base 路径；找不到则 fallback 到 ~/miniconda3。
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_base = Path(conda_exe).resolve().parents[1]
    else:
        conda_base = Path.home() / "miniconda3"

    py_name = "python.exe" if os.name == "nt" else "python"
    py_path = conda_base / "envs" / env_name / ("Scripts" if os.name == "nt" else "bin") / py_name

    if not py_path.exists():
        raise FileNotFoundError(f"找不到 conda 环境 {env_name} 的 python: {py_path}")

    return str(py_path)


def _error_payload(
    stage: str,
    message: str,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = None,
) -> str:
    return json.dumps(
        [
            {
                "category": "drug_delivery_design",
                "error": stage,
                "message": message,
                "returncode": returncode,
                "stderr": (stderr or "").strip(),
                "stdout": (stdout or "").strip(),
                "candidate_id": None,
                "best_design_id": None,
                "smiles": None,
                "drug_properties": {},
                "delivery_system": {},
                "md_metrics": {},
                "score": {},
            }
        ],
        ensure_ascii=False,
    )


def design_drug_delivery_system(
    csv_path: Annotated[str, "ADMET筛选后的CSV文件路径"]
) -> str:
    project_root = Path(__file__).resolve().parent.parent

    preprocess_script = project_root / "data_preprocessing" / "data_preprocessing.py"
    input_csv = Path(csv_path).resolve()

    if not preprocess_script.exists():
        return _error_payload(
            "preprocess_script_missing",
            f"找不到预处理脚本: {preprocess_script}",
        )

    if not input_csv.exists():
        return _error_payload(
            "input_csv_missing",
            f"输入 CSV 不存在: {input_csv}",
        )

    try:
        df_input = pd.read_csv(input_csv)
    except Exception as e:
        return _error_payload(
            "input_csv_bad",
            f"输入 CSV 读取失败: {input_csv}; {e}",
        )

    if df_input.empty:
        return _error_payload(
            "admet_no_candidates",
            f"ADMET 筛选后无候选分子: {input_csv}",
        )

    out_dir = project_root / "delivery_pipeline" / "output" / "delivery_runs_once"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex
    work_root = out_dir / f"run_{run_id}"
    work_root.mkdir(parents=True, exist_ok=True)

    enriched_csv = work_root / f"{input_csv.stem}_enriched.csv"
    output_json = work_root / "result_once.json"

    try:
        pka_python = _find_conda_env_python("pka310")
        amber_python = _find_conda_env_python("amber")
    except Exception as e:
        return _error_payload("env_not_found", str(e))

    # ---------- 第一步：pka 环境 ----------
    try:
        pka_proc = subprocess.run(
            [
                pka_python,
                str(preprocess_script),
                "--input-csv",
                str(input_csv),
                "--output-csv",
                str(enriched_csv),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=project_root,
        )
    except subprocess.TimeoutExpired as e:
        return _error_payload(
            "pka_timeout",
            f"pka 预处理超时: {e}",
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        )
    except Exception as e:
        return _error_payload("pka_subprocess_failed", str(e))

    if pka_proc.returncode != 0:
        return _error_payload(
            "pka_failed",
            "data_preprocessing.py 执行失败",
            stdout=pka_proc.stdout,
            stderr=pka_proc.stderr,
            returncode=pka_proc.returncode,
        )

    if not enriched_csv.exists():
        return _error_payload(
            "pka_no_output",
            f"预处理完成，但没找到输出文件: {enriched_csv}",
            stdout=pka_proc.stdout,
            stderr=pka_proc.stderr,
            returncode=pka_proc.returncode,
        )

    if enriched_csv.stat().st_size == 0:
        return _error_payload(
            "pka_empty_output",
            f"预处理输出是空文件: {enriched_csv}",
            stdout=pka_proc.stdout,
            stderr=pka_proc.stderr,
            returncode=pka_proc.returncode,
        )

    try:
        df_check = pd.read_csv(enriched_csv)
    except Exception as e:
        return _error_payload(
            "pka_bad_output",
            f"预处理输出不是合法CSV: {enriched_csv}; {e}",
            stdout=pka_proc.stdout,
            stderr=pka_proc.stderr,
            returncode=pka_proc.returncode,
        )

    if df_check.empty:
        return _error_payload(
            "pka_no_candidates",
            f"预处理输出CSV无有效候选: {enriched_csv}",
            stdout=pka_proc.stdout,
            stderr=pka_proc.stderr,
            returncode=pka_proc.returncode,
        )

    # ---------- 第二步：amber 环境 ----------
    # 不直接跑 pipeline.py 文件，而是在 amber 环境里 import 后执行
    amber_runner = r'''
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
input_csv = Path(sys.argv[2]).resolve()
output_json = Path(sys.argv[3]).resolve()
work_root = Path(sys.argv[4]).resolve()

sys.path.insert(0, str(project_root))

from delivery_pipeline.pipeline import run_delivery_pipeline

result = run_delivery_pipeline(
    admet_csv_path=str(input_csv),
    config={
        "work_root": str(work_root),
    },
)

output_json.parent.mkdir(parents=True, exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False))
'''
    amber_env = os.environ.copy()
    amber_bin = str(Path(amber_python).resolve().parent)

    amber_env["PYTHONPATH"] = str(project_root) + os.pathsep + amber_env.get("PYTHONPATH", "")
    amber_env["PATH"] = amber_bin + os.pathsep + amber_env.get("PATH", "")
    amber_env["CUDA_VISIBLE_DEVICES"] = amber_env.get("CUDA_VISIBLE_DEVICES", "0")
    amber_env["OPENMM_DEFAULT_PLATFORM"] = amber_env.get("OPENMM_DEFAULT_PLATFORM", "CUDA")
    amber_env["PACKMOL_BIN"] = str(Path(amber_bin) / ("packmol.exe" if os.name == "nt" else "packmol"))

    proc = None
    amber_lines: list[str] = []
    amber_stdout = ""

    try:
        proc = subprocess.Popen(
            [
                amber_python,
                "-u",
                "-c",
                amber_runner,
                str(project_root),
                str(enriched_csv),
                str(output_json),
                str(work_root),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root,
            env=amber_env,
        )

        if proc.stdout is None:
            raise RuntimeError("amber 子进程未正确创建 stdout 管道")

        for line in proc.stdout:
            print(line, end="", flush=True)
            amber_lines.append(line)

        ret = proc.wait(timeout=7200)
        amber_stdout = "".join(amber_lines)

        if ret != 0:
            return _error_payload(
                "amber_failed",
                "delivery pipeline 执行失败",
                stdout=amber_stdout,
                stderr="",
                returncode=ret,
            )

    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                if proc.stdout is not None:
                    rest = proc.stdout.read()
                    if rest:
                        amber_lines.append(rest)
            except Exception:
                pass
        amber_stdout = "".join(amber_lines)
        return _error_payload(
            "amber_timeout",
            "amber pipeline 超时",
            stdout=amber_stdout,
            stderr="",
        )
    except Exception as e:
        amber_stdout = "".join(amber_lines)
        return _error_payload(
            "amber_subprocess_failed",
            str(e),
            stdout=amber_stdout,
            stderr="",
        )
    finally:
        if proc is not None and proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass

    if not output_json.exists():
        return _error_payload(
            "amber_no_output",
            f"pipeline 执行完成，但没找到输出 JSON: {output_json}",
            stdout=amber_stdout,
            stderr="",
            returncode=0,
        )

    # ---------- 第三步：读结果并返回 ----------
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return _error_payload(
            "result_read_failed",
            f"读取输出 JSON 失败: {e}",
            stdout=amber_stdout,
            stderr="",
            returncode=0,
        )

    if isinstance(data, dict):
        data = [data]

    delivery_items = [
        x for x in data
        if isinstance(x, dict) and x.get("category") == "drug_delivery_design"
    ]

    return json.dumps(delivery_items, ensure_ascii=False)


def find_csv_directories(root_dir):
    """
    递归查找包含hash_ligand_mapping.csv的目录
    支持任何层级的子目录
    """
    valid_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "hash_ligand_mapping.csv" in filenames:
            valid_dirs.append(dirpath)
    return valid_dirs


OUT_DIR = (BASE / ".." / "admet_output").resolve()