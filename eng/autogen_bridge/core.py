# core.py
# -*- coding: utf-8 -*-

import json
import re
import ast
import warnings
import contextlib
import functools
from pathlib import Path
import torch

# 保持与原行为一致：torch.load 默认 weights_only=False
torch.load = functools.partial(torch.load, weights_only=False)

# 统一 BASE
BASE = Path(__file__).resolve().parent

warnings.simplefilter("ignore")

@contextlib.contextmanager
def suppress_rdkit_warnings():
    """全面抑制RDKit和Boost相关警告（从原文件搬过来）"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*to-Python converter for boost::shared_ptr<.*> already registered.*",
            category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore",
            module="rdkit.*",
            category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=".*already registered; second conversion method ignored.*",
            category=RuntimeWarning
        )
        yield

def parse_json_response(response_content: str) -> dict:
    """解析JSON响应内容（使用你原来的同步版本逻辑）"""
    try:
        return json.loads(response_content)
    except Exception:
        json_match = re.search(r"\{.*\}", str(response_content), re.DOTALL)
        if json_match:
            s = json_match.group(0)
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    pass
        return {"category": "protein_target_seq", "content": str(response_content)}


# 定义 InputInfo 类，用于格式化输出
class InputInfo:
    def __init__(self, category, disease_name=None, gene_target=None, content=None, 
                 targets=None, file_path=None, file_count=None, file_paths=None):
        self.category = category
        self.disease_name = disease_name
        self.gene_target = gene_target
        self.content = content
        self.targets = targets
        self.file_path = file_path
        self.file_count = file_count  # 新增字段
        self.file_paths = file_paths  # 新增字段

    def __str__(self):
        if self.category == "admet_filter":
            return f"类别: {self.category}\n" \
                  f"输出文件数: {self.file_count}\n" \
                  f"文件路径: {', '.join(self.file_paths) if self.file_paths else '无'}"
        elif self.category == "disease_protein_targets":
            return f"类别: {self.category}\n疾病名称: {self.disease_name}\n靶点数量: {len(self.targets) if self.targets else 0}\n文件路径: {self.file_path}"
        elif self.category == "protein_target_seq":
            if self.file_path and "成功生成" in self.content:
                return f"类别: {self.category}\n内容: {self.content}\n文件路径: {self.file_path}"
            elif self.file_path:
                return f"类别: {self.category}\n基因靶点: {self.gene_target}\n蛋白质序列长度: {len(self.content) if self.content else 0} 字符\n文件路径: {self.file_path}"
            else:
                return f"类别: {self.category}\n内容: {self.content}"
        else:
            return f"类别: {self.category}\n内容: {self.content}"