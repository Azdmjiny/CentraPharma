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
接收到ADMET筛选后的CSV路径后，你需要调用design_drug_delivery_system工具函数，对候选药物进行分析，并按照指定结果结构填写输出内容。

返回格式必须为严格的JSON数组，数组中每个元素代表一个候选药物的递送方案。你需要根据工具返回结果，将内容填写到以下字段中：

1. "category"：固定填写为 "drug_delivery_design"
2. "smiles"：填写该候选药物的SMILES码
3. "drug_properties"：填写药物基础性质，包括：
   - "MW"：分子量
   - "logP"：脂水分配系数
   - "tPSA"：拓扑极性表面积
   - "HBA"：氢键受体数
   - "HBD"：氢键供体数
   - "RotB"：可旋转键数
   - "QED"：类药性评分
   - "BBB"：血脑屏障相关预测结果
   - "hERG"：hERG毒性预测结果
   - "AMES"：Ames致突变性预测结果
   - "Caco2"：Caco-2渗透性预测结果
   - "pKa"、"logD74"、"Tm"、"Solubility"：若无结果可填写 null
4. "delivery_system"：填写递送系统设计信息，包括：
   - "type"：递送系统类型
   - "material"：递送材料
   - "targeting_ligand"：靶向配体，无则填 null
   - "size_nm"：粒径，单位nm
   - "zeta_mv"：zeta电位，单位mV
   - "drug_loading"：载药量
   - "packmol_ok"：Packmol构建是否成功
   - "packmol_pdb"：Packmol输出的PDB文件路径
   - "openmm_min_pdb"：OpenMM最小化后的PDB文件路径
5. "md_metrics"：填写分子模拟或包装评价结果，包括：
   - "mode"：模拟模式
   - "openmm_min_pdb"：对应的最小化结构文件路径
   - "packaging_metrics"：
     - "delta_min_A"：最小距离相关指标
     - "drug_radius_mean_A"：药物分布半径平均值
     - "drug_nn_dist_mean_A"：药物最近邻距离平均值
6. "score"：填写综合评分结果，包括：
   - "total"：总评分
   - "breakdown"：
     - "S_material"：材料评分
     - "S_structure"：结构评分
     - "S_md"：模拟评分
     - "S_qed"：类药性评分
7. "best_design_id"：填写最佳递送方案ID
8. "candidate_id"：填写候选化合物ID

返回格式示例如下：
[
  {
    "category": "drug_delivery_design",
    "smiles": "药物SMILES码",
    "drug_properties": {...},
    "delivery_system": {...},
    "md_metrics": {...},
    "score": {...},
    "best_design_id": "最佳设计ID",
    "candidate_id": "候选化合物ID"
  }
]

必须：
1. 调用提供的工具函数 design_drug_delivery_system
2. 返回严格的JSON数组格式
3. 使用双引号
4. 按上述字段含义填写内容，不得遗漏字段
5. 禁止额外解释
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

