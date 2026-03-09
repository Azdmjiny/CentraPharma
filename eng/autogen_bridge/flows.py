# flows.py
# -*- coding: utf-8 -*-

import os
import json
import asyncio

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from core import InputInfo, parse_json_response
from tools import batch_generate_ligands, generate_ligands

def _push(messages: list[dict], role: str, content: str):
    messages.append({"role": role, "content": content})


#############################################################
# 新增：执行递送方案设计，并显示
#############################################################
async def process_post_admet_delivery(file_paths, agents: dict, messages: list[dict]):
    for csv_file in file_paths:
        if not os.path.exists(csv_file):
            _push(messages, "assistant", f"未找到ADMET结果文件：`{csv_file}`")
            continue

        try:
            delivery_response = await agents["post_admet_delivery_agent"].on_messages(
                [TextMessage(content=f"ADMET筛选后的CSV路径：{csv_file}", source="user")],
                CancellationToken()
            )

            raw_content = delivery_response.chat_message.content
            print("===== RAW DELIVERY CONTENT =====")
            print(raw_content)

            delivery_result = parse_json_response(raw_content)
            print("===== PARSED DELIVERY RESULT =====")
            print(delivery_result)

            if isinstance(delivery_result, dict):
                delivery_result = [delivery_result]

            if not isinstance(delivery_result, list):
                _push(
                    messages,
                    "assistant",
                    f"递送系统设计返回格式错误，期望 JSON 数组，实际内容：\n```json\n{raw_content}\n```"
                )
                continue

            # 只保留真正的 drug_delivery_design
            valid_items = []
            for item in delivery_result:
                if not isinstance(item, dict):
                    continue
                if item.get("category") != "drug_delivery_design":
                    continue
                valid_items.append(item)

            if not valid_items:
                _push(
                    messages,
                    "assistant",
                    f"未解析到有效的递送系统设计结果。原始返回：\n```json\n{raw_content}\n```"
                )
                continue

            for item in valid_items:
                candidate_id = item.get("candidate_id", "未知")
                best_design_id = item.get("best_design_id", "未知")
                smiles = item.get("smiles", "未知")

                drug_properties = item.get("drug_properties") or {}
                delivery_system = item.get("delivery_system") or {}
                md_metrics = item.get("md_metrics") or {}
                packaging_metrics = md_metrics.get("packaging_metrics") or {}
                score = item.get("score") or {}
                score_breakdown = score.get("breakdown") or {}

                content = (
                    f"**递送系统设计结果**\n"
                    f"- candidate_id: `{candidate_id}`\n"
                    f"- best_design_id: `{best_design_id}`\n"
                    f"- smiles: `{smiles}`\n"
                    f"- category: `{item.get('category')}`\n"
                    f"- drug_properties: `{json.dumps(drug_properties, ensure_ascii=False)}`\n"
                    f"- delivery_system: `{json.dumps(delivery_system, ensure_ascii=False)}`\n"
                    f"- md_metrics: `{json.dumps(md_metrics, ensure_ascii=False)}`\n"
                    f"- packaging_metrics: `{json.dumps(packaging_metrics, ensure_ascii=False)}`\n"
                    f"- score_total: `{score.get('total', None)}`\n"
                    f"- score_breakdown: `{json.dumps(score_breakdown, ensure_ascii=False)}`"
                )
                _push(messages, "assistant", content)

        except Exception as e:
            _push(
                messages,
                "assistant",
                f"处理递送系统设计失败：`{csv_file}`\n错误信息：`{str(e)}`"
            )
            
# 定义执行流程
async def execute_disease_flow(disease_name, agents: dict, messages: list[dict]):
    """执行疾病名称全流程"""
    # 1. 疾病到靶点
    response = await agents["disease_to_protein_agent"].on_messages(
        [TextMessage(content=f"将疾病 {disease_name} 转换为蛋白靶点序列", source="user")],
        CancellationToken()
    )
    result = parse_json_response(response.chat_message.content)

    if isinstance(result, dict) and result.get("error"):
        _push(messages, "assistant", f"❌ disease_to_protein_sequences error: {result['error']}")
        return

    input_info = InputInfo(
        category=result.get("category"),
        disease_name=result.get("disease_name"),
        targets=result.get("targets"),
        file_path=result.get("file_path")
    )
    _push(messages, "assistant", f"**疾病靶点识别结果**\n{str(input_info)}")
    
    # 2. 批量生成配体
    if result.get("file_path") and os.path.exists(result.get("file_path")):
        ligand_results = await asyncio.get_event_loop().run_in_executor(
            None, batch_generate_ligands, result.get("file_path")
        )
        
        for idx, ligand_result in enumerate(ligand_results):
            ligand_input_info = InputInfo(
                category=ligand_result.get("category"),
                content=ligand_result.get("content"),
                file_path=ligand_result.get("file_path")
            )
            _push(messages, "assistant", f"**配体生成结果 {idx+1}**\n{str(ligand_input_info)}")
            
            # 3. ADMET筛选
            if ligand_result.get("file_path") and os.path.exists(ligand_result.get("file_path")):
                filter_response = await agents["admet_filter_agent"].on_messages(
                    [TextMessage(content=f"对以下目录执行ADMET筛选：{ligand_result.get('file_path')}", source="user")],
                    CancellationToken()
                )
                filter_result = parse_json_response(filter_response.chat_message.content)
                
                # 确保返回的结果是字典形式
                if isinstance(filter_result, dict):
                    filter_input_info = InputInfo(
                        category=filter_result.get("category"),
                        file_count=filter_result.get("file_count"),
                        file_paths=filter_result.get("file_paths")
                    )
                    _push(messages, "assistant", f"**ADMET筛选结果**\n{str(filter_input_info)}")
                    # >>> 新增：调用递送方案设计 <<<
                    if filter_result.get("file_paths"):
                        await process_post_admet_delivery(filter_result["file_paths"], agents, messages)

async def execute_gene_flow(gene_name: str, agents: dict, messages: list[dict]):
    """执行基因全流程（无 Streamlit 依赖）"""

    # 1) 基因 -> 蛋白
    response = await agents["gene_to_protein_agent"].on_messages(
        [TextMessage(content=f"将基因 {gene_name} 转换为蛋白质靶点序列", source="user")],
        CancellationToken()
    )
    result = parse_json_response(response.chat_message.content)

    if isinstance(result, dict) and result.get("error"):
        _push(messages, "assistant", f"❌ gene_target_to_protein_sequence error: {result['error']}")
        return

    input_info = InputInfo(
        category=result.get("category"),
        gene_target=result.get("gene_target"),
        content=result.get("content"),
        file_path=result.get("file_path")
    )
    _push(messages, "assistant", f"**基因翻译结果**\n{str(input_info)}")

    # 2) 生成配体（你原逻辑是直接调用 tools.generate_ligands）
    if result.get("file_path") and os.path.exists(result.get("file_path")):
        ligand_result = await asyncio.get_event_loop().run_in_executor(
            None, generate_ligands, result.get("content")
        )
        ligand_dict = json.loads(ligand_result)

        ligand_input_info = InputInfo(
            category=ligand_dict.get("category"),
            content=ligand_dict.get("content"),
            file_path=ligand_dict.get("file_path")
        )
        _push(messages, "assistant", f"**配体生成结果**\n{str(ligand_input_info)}")

        # 3) ADMET 筛选
        if ligand_dict.get("file_path") and os.path.exists(ligand_dict.get("file_path")):
            print("[FLOW] ADMET工具返回，开始解析结果")
            filter_response = await agents["admet_filter_agent"].on_messages(
                [TextMessage(content=f"对以下目录执行ADMET筛选：{ligand_dict.get('file_path')}", source="user")],
                CancellationToken()
            )
            filter_result = parse_json_response(filter_response.chat_message.content)
            print("[FLOW] ADMET结果解析完成:", filter_result)
            filter_input_info = InputInfo(
                category=filter_result.get("category"),
                file_count=filter_result.get("file_count"),
                file_paths=filter_result.get("file_paths")
            )
            _push(messages, "assistant", f"**ADMET筛选结果**\n{str(filter_input_info)}")

            # 4) 递送方案设计
            if filter_result.get("file_paths"):
                print("[FLOW] 开始递送设计")
                await process_post_admet_delivery(filter_result["file_paths"], agents, messages)
                print("[FLOW] 递送设计完成")


async def execute_protein_flow(protein_seq: str, agents: dict, messages: list[dict]):
    """执行蛋白质序列全流程（flows 版本：不依赖 Streamlit）"""
    # 1. 生成配体（generate_ligands 在 tools.py）
    ligand_result = await asyncio.get_event_loop().run_in_executor(
        None, generate_ligands, protein_seq
    )
    ligand_dict = json.loads(ligand_result)

    ligand_input_info = InputInfo(
        category=ligand_dict.get("category"),
        content=ligand_dict.get("content"),
        file_path=ligand_dict.get("file_path")
    )
    _push(messages, "assistant", f"**配体生成结果**\n{str(ligand_input_info)}")

    # 2. ADMET 筛选（走 agent 工具调用）
    ligand_path = ligand_dict.get("file_path")
    if ligand_path and os.path.exists(ligand_path):
        filter_response = await agents["admet_filter_agent"].on_messages(
            [TextMessage(content=f"对以下目录执行ADMET筛选：{ligand_path}", source="user")],
            CancellationToken()
        )

        # core.parse_json_response 是同步函数，不要 await
        filter_result = parse_json_response(filter_response.chat_message.content)

        filter_input_info = InputInfo(
            category=filter_result.get("category"),
            file_count=filter_result.get("file_count"),
            file_paths=filter_result.get("file_paths")
        )
        _push(messages, "assistant", f"**ADMET筛选结果**\n{str(filter_input_info)}")

        # 3. 递送方案设计（把 agents/messages 传下去）
        if filter_result.get("file_paths"):
            await process_post_admet_delivery(filter_result["file_paths"], agents, messages)


async def execute_carrier_flow(design_content: str, agents: dict, messages: list[dict]):
    """执行递送系统设计流程（纳米载体设计）"""
    response = await agents["nano_carrier_agent"].on_messages(
        [TextMessage(content=f"设计针对{design_content}的中枢神经系统纳米载体", source="user")],
        CancellationToken()
    )

    result = parse_json_response(response.chat_message.content)

    if isinstance(result, dict) and result.get("error"):
        _push(messages, "assistant", f"❌ nano_carrier_design error: {result['error']}")
        return

    input_info = InputInfo(
        category=result.get("category"),
        content=result.get("content"),
        file_path=result.get("file_path")
    )
    _push(messages, "assistant", f"**纳米载体设计结果**\n{str(input_info)}")

async def handle_query(user_input: str, agents: dict, messages: list[dict]):
    """处理用户查询的核心逻辑（不依赖 Streamlit，全靠 agents + messages）"""
    # 1. 分类任务
    classification_response = await agents["classifier_agent"].on_messages(
        [TextMessage(content=f"请帮我识别以下输入的类型：{user_input}", source="user")],
        CancellationToken()
    )
    classification_result = parse_json_response(classification_response.chat_message.content)
    category = classification_result.get("category", "")
    content = classification_result.get("content", "")

    # 2. 根据分类执行完整流程
    if category == "disease_name":
        await execute_disease_flow(content, agents, messages)
    elif category == "gene_target":
        await execute_gene_flow(content, agents, messages)
    elif category == "protein_target_seq":
        await execute_protein_flow(content.strip(), agents, messages)
    elif category == "nano_carrier_design":
        await execute_carrier_flow(content, agents, messages)
    else:
        _push(messages, "assistant", f"暂不支持处理 '{category}' 类型的输入")

