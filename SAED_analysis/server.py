import os
import json
import asyncio
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from typing import Optional
import openai

# 你的 4 个工具

from tools.predict import pre_symmetry
from tools.dp_retrieval import retrieve_for_two_images
from tools.daizhou import match_experiment_with_simulation
from tools.dp_add import add_images_to_database

app = FastAPI(title="电镜分析AI（无ADK版）")
os.makedirs("uploads", exist_ok=True)

# DeepSeek 客户端
client = openai.AsyncOpenAI(
    api_key="sk-4845a18564ca45be8e84e6649ce9604d",
    base_url="https://api.deepseek.com"
)

# 🔥 修复：工具定义格式（完全符合 OpenAI 规范）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pre_symmetry",
            "description": "空间群预测，支持1或2张电镜图",
            "parameters": {
                "type": "object",
                "properties": {
                    "image1": {"type": "string", "description": "图片1路径"},
                    "image2": {"type": "string", "description": "图片2路径（可选）"}
                },
                "required": ["image1"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_for_two_images",
            "description": "双图相似检索",
            "parameters": {
                "type": "object",
                "properties": {
                    "img1": {"type": "string"},
                    "img2": {"type": "string"}
                },
                "required": ["img1","img2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_experiment_with_simulation",
            "description": "实验图像与CIF模拟图谱匹配",
            "parameters": {
                "type": "object",
                "properties": {
                    "exp_img_input": {"type": "string"},
                    "cif_input": {"type": "string"}
                },
                "required": ["exp_img_input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_images_to_database",
            "description": "图像入库",
            "parameters": {
                "type": "object",
                "properties": {
                    "images_input": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "sg": {"type": "number"},
                    "mid": {"type": "number"}
                },
                "required": ["images_input"]
            }
        }
    }
]

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>电镜结构AI分析</title>
    <style>
        body{max-width:1000px;margin:30px auto;padding:20px;font-family:Arial}
        .chat{border:1px solid #ddd;padding:20px;height:550px;overflow-y:auto;border-radius:10px;background:#fafafa}
        .msg{margin:10px 0;padding:12px 16px;border-radius:10px;max-width:80%}
        .user{background:#1677ff;color:white;margin-left:auto}
        .bot{background:#eee}
        .upload{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
        .input-box{display:flex;gap:10px}
        #msg{flex:1;padding:10px;border-radius:6px;border:1px solid #ddd}
        button{padding:10px 20px;background:#1677ff;color:white;border:none;border-radius:6px}
    </style>
</head>
<body>
    <h2>🔬 电镜结构AI分析（无ADK稳定版）</h2>
    <div class="upload">
        <div>图1：<input type="file" id="img1"></div>
        <div>图2：<input type="file" id="img2"></div>
        <div>CIF：<input type="file" id="cif"></div>
    </div>
    <div class="chat" id="chat"></div>
    <div class="input-box">
        <input type="text" id="msg" placeholder="输入指令">
        <button onclick="send()">发送</button>
    </div>

<script>
async function send(){
    const msg = document.getElementById("msg").value;
    const img1 = document.getElementById("img1").files[0];
    const img2 = document.getElementById("img2").files[0];
    const cif = document.getElementById("cif").files[0];
    const chat = document.getElementById("chat");

    chat.innerHTML += `<div class='msg user'>${msg}</div>`;
    document.getElementById("msg").value = "";

    const fd = new FormData();
    fd.append("msg", msg);
    if(img1) fd.append("img1", img1);
    if(img2) fd.append("img2", img2);
    if(cif) fd.append("cif", cif);

    const res = await fetch("/chat", {method:"POST", body:fd});
    const data = await res.json();
    chat.innerHTML += `<div class='msg bot'>${data.ans}</div>`;
    chat.scrollTop = chat.scrollHeight;
}
</script>
</body>
</html>"""

def save(uf: Optional[UploadFile]) -> Optional[str]:
    if not uf: return None
    p = f"uploads/{uuid.uuid4()}_{uf.filename}"
    with open(p, "wb") as f:
        f.write(uf.file.read())
    return p

async def call_tool(name, args):
    try:
        if name == "pre_symmetry":
            return await pre_symmetry(**args)
        if name == "retrieve_for_two_images":
            return await retrieve_for_two_images(**args)
        if name == "match_experiment_with_simulation":
            return await match_experiment_with_simulation(**args)
        if name == "add_images_to_database":
            return add_images_to_database(**args)
        return "未知工具"
    except Exception as e:
        return f"工具执行错误：{str(e)}"

@app.post("/chat")
async def chat(
    msg: str = Form(...),
    img1: Optional[UploadFile] = File(None),
    img2: Optional[UploadFile] = File(None),
    cif: Optional[UploadFile] = File(None)
):
    p1 = save(img1)
    p2 = save(img2)
    c = save(cif)

    info = f"用户消息：{msg}\n图1路径：{p1}\n图2路径：{p2}\nCIF路径：{c}"

    messages = [
        {"role": "system", "content": "你是电镜SAED结构分析专家，调用工具完成预测、检索、匹配、入库。"},
        {"role": "user", "content": info}
    ]

    res = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS
    )

    ans = ""
    choice = res.choices[0]

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            func_name = tc.function.name
            args = json.loads(tc.function.arguments)
            ans += f"🔧 正在调用工具：{func_name}\n"
            ans += f"参数：{args}\n"
            tool_result = await call_tool(func_name, args)
            ans += f"✅ 结果：\n{tool_result}\n\n"
    else:
        ans = choice.message.content

    return {"ans": ans}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)