import os
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.runners import InMemoryRunner

import nest_asyncio
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# ===================== 模型密钥 =====================
os.environ['DEEPSEEK_API_KEY'] = "sk-4845a18564ca45be8e84e6649ce9604d"

# ===================== MCP SSE 连接 =====================
server_url = "http://127.0.0.1:50002/sse"
sse_params = SseServerParams(url=server_url)
toolset = McpToolset(connection_params=sse_params)

# ===================== Agent 定义 =====================
model = LiteLlm(model="deepseek/deepseek-chat")

root_agent = LlmAgent(
    name="EM_Atomic_Structure_Recognition_Agent",
    model=model,
    description="电子显微镜原子结构智能分析助手",
    instruction=(
        "你是电子显微镜与晶体结构分析专家。"
        "用户提供图片路径或文件信息时，主动调用可用工具：空间群预测、图谱检索、实验模拟匹配、图像入库。"
        "调用工具前检查必填参数，缺失则礼貌询问用户。"
        "输出清晰、专业、结构化结果。"
    ),
    tools=[toolset]
)

# 全局唯一 Runner，给网页调用
runner = InMemoryRunner(root_agent)