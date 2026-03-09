# streamlit_app.py
# -*- coding: utf-8 -*-

import asyncio
import streamlit as st
import nest_asyncio

from agents import build_agents
from flows import handle_query

nest_asyncio.apply()


# 设置页面标题和布局
st.set_page_config(page_title="中枢智药——药物设计+药物递送多智能体平台", layout="wide")
st.title("🔬 中枢智药——药物设计+药物递送多智能体平台")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "您好！我是智能助手，支持以下任务：\n"
                    "- 基因靶点 → 蛋白质序列\n"
                    "- 疾病名称 → 靶点筛选\n"
                    "- 蛋白质序列 → 生成配体\n"
                    "- 配体 → ADMET筛选\n"
                    "- 纳米载体设计\n"
                    "请告诉我您的需求。"
    })


if "agents" not in st.session_state:
    st.session_state["agents"] = build_agents()

def display_messages(start_idx: int = 0):
    for message in st.session_state.messages[start_idx:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
display_messages()


# 获取用户输入
user_input = st.chat_input("请输入您的请求，例如：将疾病 Alzheimer 转换为蛋白靶点序列")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 记录：后续新增消息从哪个下标开始
    start_idx = len(st.session_state.messages)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(handle_query(
        user_input,
        st.session_state["agents"],
        st.session_state.messages
    ))
    display_messages(start_idx)