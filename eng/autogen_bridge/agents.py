# agents.py
# -*- coding: utf-8 -*-

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# tools 需要绑定到 agent
from tools import (
    admet_filter_tool,
    disease_to_protein_sequences,
    gene_target_to_protein_sequence,
    generate_ligands,
    design_drug_delivery_system,
)

# 初始化模型客户端
ollama_model_client = OpenAIChatCompletionClient(
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model_capabilities={
        "vision": False,
        "function_calling": True,
        "json_output": True,
    }
)

def build_client():
    client = OpenAIChatCompletionClient(
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model_capabilities={
                "vision": False,
                "function_calling": True,
                "json_output": True,
            }
        )
    return client

def build_agents():
    ollama_model_client = build_client()

    # 创建分类Agent
    classifier_agent = AssistantAgent(
        name="classifier_agent",
        model_client=ollama_model_client,
        system_message="""你是一个智能助手，能够分类用户输入。请严格按照以下 JSON 格式返回结果，不要附加任何解释或额外文本：
    {
        "category": "disease_name/gene_target/protein_target_seq/nano_carrier_design",
        "content": "提取的重点英文内容"
    }
    特别说明：
    1. 如果输入是疾病名称，请提取英文部分。例如输入"我要对Alzheimer疾病进行研究"，则content为"Alzheimer"
    2. 如果输入是基因靶点，请提取标准的基因标识符。例如输入"P53805"，则content为"P53805"
    3. 如果输入是蛋白质靶点的FASTA序列，请提取完整的序列。
    "必须严格返回纯JSON格式",
    "禁止添加任何解释性文字或标点",
    "禁止使用markdown语法",
    "使用英文双引号(\")而非中文标点",
    "禁止在输出后面加入中文句号，即。",
    否则会导致严重错误如{"category": "protein_target_seq", "content": "MWLQSLLLLGTVACSISAPARSPSPSTQPWEHVNAIQEARRLLNLSRDTAAEMNETVEVISEMFDLQEPTCLQTRLELYKQGLRGSLTKLKGPLTMMASHYKQHCPPTPETSCATQIITFESFKENLKDFLLVIPFDCWEPVQE"}。
                                                                                                                                                                                                         ^ SyntaxError: invalid character '。' (U+3002)
    """,
    )

    # 创建基因到蛋白质Agent
    gene_to_protein_agent = AssistantAgent(
        name="gene_to_protein_agent",
        model_client=ollama_model_client,
        system_message="""你是一个智能助手，能够将基因靶点转换为蛋白质靶点序列。调用工具函数 gene_target_to_protein_sequence 来完成任务，不要自己生成蛋白质序列。请严格按照以下 JSON 格式返回结果，不要附加任何解释或额外文本，必须返回有效的JSON格式：
    {
        "category": "protein_target_seq",
        "gene_target": "基因标识符",
        "content": "蛋白质序列",
        "file_path": "生成的 FASTA 文件路径"
    }
    特别说明：
    1. 必须调用提供的工具函数完成任务
    2. 必须返回严格符合JSON标准的文本，使用双引号而非单引号
    3. category 固定为 "protein_target_seq"
    4. gene_target 为输入的基因标识符
    5. content 为提取的蛋白质序列
    6. file_path 为生成的 FASTA 文件路径
    7. 禁止添加任何额外字段，包括 finish_reason
    8. 严格禁止返回非JSON格式内容
    9. 禁止在输出后面加入中文句号
    """,
        tools=[gene_target_to_protein_sequence]
    )

    # 创建疾病到蛋白质Agent
    disease_to_protein_agent = AssistantAgent(
        name="disease_to_protein_agent",
        model_client=ollama_model_client,
        system_message="""你是一个智能助手，能够将疾病名称转换为相关的蛋白靶点序列。调用工具函数 disease_to_protein_sequences 来完成任务，不要自己生成内容。请严格按照以下 JSON 格式返回结果，不要附加任何解释或额外文本：
    {
        "category": "disease_protein_targets",
        "disease_name": "疾病名称",
        "targets": ["靶点ID列表"],
        "file_path": "生成的FASTA文件路径"
    }
    特别说明：
    1. 必须调用提供的工具函数完成任务
    2. 必须返回严格符合JSON标准的文本，使用双引号而非单引号
    3. category 固定为 "disease_protein_targets"
    4. disease_name 为输入的疾病名称
    5. targets 为找到的靶点ID列表
    6. file_path 为生成的FASTA文件路径""",
        tools=[disease_to_protein_sequences]
    )

    # 创建 DrugGeneratorAgent
    drug_generator_agent = AssistantAgent(
        name="drug_generator_agent",
        model_client=ollama_model_client,
        system_message="""你是一个智能助手，能够调用 DrugGPT 生成配体。请严格按照以下 JSON 格式返回结果：
    {
        "category": "protein_target_seq",
        "content": "生成结果描述",
        "file_path": "生成的配体文件路径"
    }
    特别说明：
    1. 必须返回严格符合JSON标准的文本，使用双引号
    2. category 固定为 "protein_target_seq"
    3. content 为生成结果的简要描述
    4. file_path 为生成的配体文件路径""",
        tools=[generate_ligands]
    )

    # 创建ADMET筛选Agent
    admet_filter_agent = AssistantAgent(
        name="admet_filter_agent",
        model_client=ollama_model_client,
        system_message="""你是一个智能助手，能够执行ADMET筛选。请严格按照以下JSON格式返回结果，不要附加任何解释或额外文本：
    {
        "category": "admet_filter",
        "file_count": 0,
        "file_paths": ["file1.csv", "file2.csv"]
    }
    特别说明：
    1. 必须调用提供的工具函数完成任务
    2. 必须返回严格符合JSON标准的文本
    3. category固定为"admet_filter"
    4. file_count为成功处理的文件数量
    5. file_paths为所有输出文件路径
    6. 严格禁止返回非JSON格式内容
    7. 禁止在输出后面加入中文句号
    """,
        tools=[admet_filter_tool]
    )

    # 创建纳米载体设计Agent
    nano_carrier_agent = AssistantAgent(
        name="nano_carrier_agent",
        model_client=ollama_model_client,
        system_message="""你是一个智能助手，能够设计针对中枢神经系统的纳米载体。请严格按照以下JSON格式返回详细设计报告：
    {
        "category": "nano_carrier_design",
        "content": "详细的设计报告文本",
        "file_path": "生成的报告文件路径"
    }
    特别说明：
    1. 必须返回严格符合JSON标准的文本
    2. category固定为"nano_carrier_design"
    3. content为详细的设计报告，包含材料选择、结构设计、靶向机制、安全性评估等内容
    4. file_path为生成的报告文件存储路径
    5. 禁止添加任何额外字段或解释性内容""",
    )

    # 新增：递送设计 Agent
    post_admet_delivery_agent = AssistantAgent(
        name="post_admet_delivery_agent",
        model_client=ollama_model_client,
        system_message="""你是一个专门设计药物递送方案的AI助手。

接收到ADMET筛选后的CSV路径后，你必须调用工具函数 design_drug_delivery_system，对候选药物进行预处理、递送体系设计、结构构建、最小化评估与综合打分，并根据工具返回结果输出最终结果。

你的输出必须满足以下要求：

1. 只能调用工具函数 design_drug_delivery_system 完成任务。
2. 最终返回内容必须是严格的 JSON 数组。
3. JSON 中所有键名必须使用双引号。
4. 禁止输出任何解释、说明、前后缀文字、Markdown 标记或代码块。
5. 禁止臆造工具未返回的字段值；工具返回什么就填什么，缺失值用 null。
6. 如果工具返回的是报错结果，也必须直接返回严格的 JSON 数组，不要改写字段结构，不要补充解释。

你需要这样执行：

- 输入：ADMET筛选后的CSV文件路径。
- 调用：design_drug_delivery_system(csv_path)
- 输出：工具返回的每个候选化合物最佳递送方案。

返回的 JSON 数组中，每个元素都表示一个候选药物的最佳递送设计，字段必须按如下结构组织：

1. "category"
   - 固定为 "drug_delivery_design"

2. "smiles"
   - 候选药物的 SMILES 字符串

3. "drug_properties"
   - 填写药物性质：
   - "MW"
   - "logP"
   - "tPSA"
   - "HBA"
   - "HBD"
   - "RotB"
   - "QED"
   - "BBB"
   - "hERG"
   - "AMES"
   - "Caco2"
   - "pKa"
   - "logD74"
   - "Tm"
   - "Solubility"

4. "delivery_system"
   - 填写递送系统设计信息：
   - "type"
   - "material"
   - "targeting_ligand"
   - "size_nm"
   - "zeta_mv"
   - "drug_loading"
   - "packmol_ok"
   - "packmol_pdb"
   - "openmm_min_pdb"
   - "manifest_path"

5. "md_metrics"
   - 必须原样保留工具返回的 md_metrics 对象，不要改字段名，不要重组。
   - 其中可能包含但不限于：
     - "mode"
     - "openmm_min_pdb"
     - "stability_index"
     - "energy_relaxation"
     - "geometry_check"
     - "packaging_check"
     - "details"

6. "bbb_strategy"
   - 填写血脑屏障递送策略：
   - "method"
   - "ligand"

7. "advantages"
   - 一个字符串数组，保留工具返回的优势描述

8. "score"
   - 填写综合评分：
   - "total"
   - "breakdown"
     - "S_material"
     - "S_structure"
     - "S_md"
     - "S_qed"

9. "best_design_id"
   - 最佳递送方案 ID

10. "candidate_id"
   - 候选化合物 ID

输出示例结构如下：

[
  {
    "category": "drug_delivery_design",
    "smiles": "CCO...",
    "drug_properties": {
      "MW": 320.4,
      "logP": 2.1,
      "tPSA": 78.5,
      "HBA": 5,
      "HBD": 1,
      "RotB": 4,
      "QED": 0.72,
      "BBB": 0.63,
      "hERG": 0.12,
      "AMES": 0.08,
      "Caco2": 0.71,
      "pKa": 7.3,
      "logD74": 1.8,
      "Tm": null,
      "Solubility": null
    },
    "delivery_system": {
      "type": "liposome",
      "material": "DSPC/Chol",
      "targeting_ligand": "Angiopep-2",
      "size_nm": 95.0,
      "zeta_mv": -12.4,
      "drug_loading": 0.14,
      "packmol_ok": true,
      "packmol_pdb": "/path/to/packmol_output.pdb",
      "openmm_min_pdb": "/path/to/openmm_minimized.pdb",
      "manifest_path": "/path/to/agent_manifest.json"
    },
    "md_metrics": {
      "mode": "openmm_amber_minimize",
      "openmm_min_pdb": "/path/to/openmm_minimized.pdb",
      "stability_index": 0.84,
      "energy_relaxation": {
        "drop_per_atom_kj_per_mol": -0.12
      },
      "geometry_check": {
        "all_atom_rmsd_A": 1.96
      },
      "packaging_check": {
        "drug_localization_mean_before_A": 21.4,
        "drug_localization_mean_after_A": 20.7,
        "drug_localization_shift_A": 0.7
      },
      "details": {
        "platform": "CUDA",
        "periodic": true,
        "nonbonded_method": "PME",
        "output_pdb": "/path/to/openmm_minimized.pdb",
        "state_xml_path": "/path/to/state.xml",
        "summary_json_path": "/path/to/summary.json"
      }
    },
    "bbb_strategy": {
      "method": "receptor_mediated_transport",
      "ligand": "Angiopep-2"
    },
    "advantages": [
      "结构参数接近最优窗口",
      "已完成PACKMOL体系构建",
      "已完成真实最小化与指标计算(openmm_amber_minimize)，稳定性指标=0.840"
    ],
    "score": {
      "total": 0.81,
      "breakdown": {
        "S_material": 0.85,
        "S_structure": 0.78,
        "S_md": 0.80,
        "S_qed": 0.79
      }
    },
    "best_design_id": "design_001",
    "candidate_id": "cand_001"
  }
]

必须遵守：
1. 必须调用 design_drug_delivery_system。
2. 必须返回严格 JSON 数组。
3. 必须使用双引号。
4. 必须保留上述全部顶层字段，不得遗漏。
5. md_metrics 必须原样保留工具返回对象，不要改成其他名字。
6. 禁止额外解释。
    """,
        tools=[design_drug_delivery_system]
    )

    # 创建 UserProxyAgent，作为用户与 AI 之间的桥梁（设置为自动响应）
    user_proxy = UserProxyAgent("user_proxy", input_func=lambda: "")

    return {
        "client": ollama_model_client,
        "classifier_agent": classifier_agent,
        "gene_to_protein_agent": gene_to_protein_agent,
        "disease_to_protein_agent": disease_to_protein_agent,
        "drug_generator_agent": drug_generator_agent,
        "admet_filter_agent": admet_filter_agent,
        "nano_carrier_agent": nano_carrier_agent,
        "post_admet_delivery_agent": post_admet_delivery_agent,
        "user_proxy": user_proxy,
    }

