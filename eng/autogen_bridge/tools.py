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

# 完全替换admet_filter_tool函数
def admet_filter_tool(input_dir: str) -> dict:
    """执行ADMET筛选的核心工具函数"""
    try:
        # 验证输入目录有效性
        if not os.path.isdir(input_dir):
            raise ValueError(f"无效的输入目录: {input_dir}")
        
        # 初始化ADMET模型
        model = get_admet_model()
        
        # 设置输出目录
        OUT_DIR = (BASE / ".." / "output").resolve()
        os.makedirs(OUT_DIR, exist_ok=True)
        
        # 查找有效目录
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
        
        # 处理每个有效目录
        for dirpath in valid_dirs:
            # 构建文件路径
            in_csv = os.path.join(dirpath, "hash_ligand_mapping.csv")
            
            # 生成唯一输出文件名
            relative_path = os.path.relpath(dirpath, start=input_dir)
            safe_path = re.sub(r'[^\w]', '_', relative_path)
            out_csv = os.path.join(OUT_DIR, f"{safe_path}_filtered.csv")
            
            try:
                # 读取SMILES数据
                df = pd.read_csv(in_csv, header=None, names=['hash', 'smiles'])
                smiles_list = df['smiles'].tolist()
                
                # 执行ADMET预测
                preds = model.predict(smiles=smiles_list)
                preds_df = pd.DataFrame(preds)
                
                # 添加SMILES列
                preds_df['smiles'] = smiles_list
                
                # 应用筛选条件（完全复用test.py的筛选条件）
                mask = (
                    (preds_df["BBB_Martins"] >= 0.40) &  # 放宽至40%
                    (preds_df["QED"] >= 0.35) &         # 降低至35%
                    (preds_df["logP"].between(0.5, 4.0)) &  # 扩展至0.5-4.0
                    (preds_df["tpsa"] <= 120) &         # 放宽至120
                    (preds_df["hERG"] <= 0.5) &         # 保持不变
                    (preds_df["AMES"] <= 0.5)           # 保持不变
                )
                
                # 应用筛选条件
                filtered = preds_df[mask]
                
                # 保存筛选结果
                filtered.to_csv(out_csv, index=False)
                
                # 记录处理结果
                results.append(f"{relative_path}: 筛选完成，共 {len(filtered)} 条分子 -> {out_csv}")
                output_files.append(out_csv)
                
            except Exception as e:
                results.append(f"{relative_path}: 处理失败 - {str(e)}")
                continue
        
        # 返回标准化结果
        return {
            "category": "admet_filter",
            "file_count": len(output_files),
            "file_paths": output_files,
            "details": results
        }
        
    except Exception as e:
        # 返回错误信息
        return {
            "category": "admet_filter",
            "error": str(e),
            "details": f"{str(e.__class__.__name__)}: {str(e)}"
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

        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                target_id, data_type = parts[0], parts[1]

                # 直接从 INDICATI 行提取靶点 ID
                if data_type == "INDICATI":
                    # 简单验证 target_id 格式（如 T12345）
                    if re.match(r'^T\d+$', target_id):
                        # 检查疾病名称是否出现在该行
                        if re.search(rf'\b{re.escape(disease_name.lower())}\b', line.lower()):
                            targets.add(target_id)
                    else:
                        print(f"[DEBUG] 跳过无效的 target_id: {target_id}")

        if not targets:
            raise ValueError(f"未找到与疾病 {disease_name} 相关的靶点。")

        # 2. 将靶点ID保存到CSV文件
        with output_file.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Target_ID'])
            for target in sorted(targets):
                writer.writerow([target])

        # 3. 从FASTA文件中查找蛋白序列
        fasta_path = (BASE / ".." / "data" / "P2-06-TTD_sequence_all.txt").resolve()
        if not fasta_path.exists():
            raise FileNotFoundError(f"找不到文件 {fasta_path}")

        sequences = {}
        current_id = None
        current_seq_parts = []

        with fasta_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('>'):
                    if current_id and current_seq_parts:
                        sequences[current_id] = ''.join(current_seq_parts)
                    header = line[1:].split('\t')[0].strip()
                    current_id = header
                    current_seq_parts = []
                else:
                    if current_id:
                        current_seq_parts.append(line.strip())

            if current_id and current_seq_parts:
                sequences[current_id] = ''.join(current_seq_parts)

        found_sequences = {tid: sequences[tid] for tid in targets if tid in sequences}

        # 4. 将找到的序列写入文件
        fasta_out = output_dir / "found_protein_sequences.fasta"
        with fasta_out.open('w', encoding='utf-8') as f:
            for tid, seq in found_sequences.items():
                f.write(f">{tid}\n")
                for i in range(0, len(seq), 80):
                    f.write(f"{seq[i:i+80]}\n")

        return {
            "category": "disease_protein_targets",
            "disease_name": disease_name,
            "targets": list(found_sequences.keys()),
            "file_path": str(fasta_out)
        }
        
    except Exception as e:
        return {
            "category": "disease_protein_targets",
            "disease_name": disease_name,
            "targets": [],
            "file_path": None,
            "error": str(e)
        }

# 定义基因靶点转换为蛋白质序列的工具函数
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
            size=10
        )

        # 获取与基因相关的 UniProt Swiss-Prot ID
        uniprot_ids = set()
        for hit in res.get('hits', []):
            swiss = hit.get('uniprot', {}).get('Swiss-Prot')
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
            "file_path": out_fasta
        }
    except Exception as e:
        return {
            "category": "protein_target_seq",
            "gene_target": gene_name,
            "content": "",
            "file_path": None,
            "error": str(e)
        }

# 定义调用 DrugGPT 生成配体的工具函数
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
            sys.executable, str(DRUGGPT_SCRIPT),
            "-p", protein_sequence,
            "-n", str(num_molecules),
            "-o", str(output_dir_min)
        ]

        # 在这里加 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        print("==== BRIDGE DEBUG ====")
        print("PYEXE:", sys.executable)
        print("torch:", torch.__version__)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("is_available:", torch.cuda.is_available())
        print("device_count:", torch.cuda.device_count())
        print("======================")

        # 执行命令
        DRUGGPT_DIR = DRUGGPT_SCRIPT.parent

        #result = subprocess.run(cmd, capture_output=True, text=True,cwd=str(DRUGGPT_DIR))
        result = subprocess.run(cmd, text=True,cwd=str(DRUGGPT_DIR))
        if result.returncode != 0:
            raise RuntimeError(f"DrugGPT 执行失败：{result.stderr}")

        return json.dumps(
            {
                "category": "protein_target_seq",
                "content": f"成功生成 {num_molecules} 个配体",
                "file_path": str(output_dir_min),
                "target_id": target_id
            },
            ensure_ascii=False
        )
    except Exception as e:
        return json.dumps(
            {
                "category": "protein_target_seq",
                "content": f"生成配体时出错：{str(e)}",
                "file_path": None,
                "target_id": target_id
            },
            ensure_ascii=False
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

def design_drug_delivery_system(
    csv_path: Annotated[str, "ADMET筛选后的CSV文件路径"]
) -> str:
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "delivery_pipeline" / "run_delivery_once.py"
    out_dir = project_root / "delivery_pipeline" / "output" / "delivery_runs_once"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex
    work_root = out_dir / f"run_{run_id}"
    work_root.mkdir(parents=True, exist_ok=True)

    output_json = work_root / "result_once.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--csv-path",
            str(Path(csv_path).resolve()),
            "--output-json",
            str(output_json),
            "--work-root",
            str(work_root),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=project_root,
    )

    if proc.returncode != 0:
        return json.dumps(
            [
                {
                    "category": "drug_delivery_design",
                    "error": "script_failed",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip(),
                    "stdout": proc.stdout.strip(),
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

    if not output_json.exists():
        return json.dumps([], ensure_ascii=False)

    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

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