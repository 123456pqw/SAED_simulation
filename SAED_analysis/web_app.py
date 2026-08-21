import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from typing import Optional, Union

# 导入你的全部工具
from tools.predict import pre_symmetry
from tools.dp_retreival import retrieve_for_two_images
from tools.daizhou import match_experiment_with_simulation
from tools.dp_add import add_images_to_database
from tools.stem2crystal import get_stem_2_crystal

app = FastAPI(title="SAED 电镜分析完整网页端")
os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SAED 智能分析平台</title>
    <style>
        body {max-width: 1100px; margin: 30px auto; padding: 20px; font-family: "Microsoft Yahei",Arial}
        .chat {border:1px solid #ddd; padding:20px; height:520px; overflow-y:auto; border-radius:10px; background:#fafafa; margin-bottom:20px;}
        .msg {margin:10px 0; padding:12px 16px; border-radius:10px; max-width:85%; white-space:pre-wrap;}
        .user {background:#1677ff; color:#fff; margin-left:auto;}
        .bot {background:#f0f2f5; color:#333; margin-right:auto;}
        .upload-row {display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:15px;}
        .input-row {display:flex; gap:10px; align-items:center;}
        #msg {flex:1; padding:10px 14px; border:1px solid #d9d9d9; border-radius:8px; font-size:14px;}
        button {padding:10px 24px; background:#1677ff; color:#fff; border:none; border-radius:8px; cursor:pointer; font-size:14px;}
        button:hover {background:#0958d9;}
        .tip {font-size:12px; color:#888; margin:8px 0;}
    </style>
</head>
<body>
    <h2>🔬 SAED 电子显微镜 | 晶体结构智能分析平台</h2>
    <div class="tip">
        指令示例：预测 / 双图检索 / 实验模拟匹配 / 图像入库
    </div>

    <div class="chat" id="chat"></div>

    <!-- 上传区域：图1 + 图2 + CIF文件 -->
    <div class="upload-row">
        <div>图像1：<input type="file" id="img1" accept="image/*,.tif,.dm3,.dm4"></div>
        <div>图像2(可选)：<input type="file" id="img2" accept="image/*,.tif,.dm3,.dm4"></div>
        <div>CIF文件(匹配用)：<input type="file" id="cifFile" accept=".cif"></div>
    </div>

    <div class="input-row">
        <input type="text" id="msg" placeholder="输入操作指令，例如：预测、双图检索、实验模拟匹配">
        <button onclick="send()">发送执行</button>
    </div>

<script>
async function send(){
    const msg = document.getElementById("msg").value.trim();
    const f1 = document.getElementById("img1").files[0];
    const f2 = document.getElementById("img2").files[0];
    const cif = document.getElementById("cifFile").files[0];
    const chat = document.getElementById("chat");

    chat.innerHTML += `<div class="msg user">${msg}</div>`;
    document.getElementById("msg").value = "";

    let fd = new FormData();
    fd.append("msg", msg);
    if(f1) fd.append("img1", f1);
    if(f2) fd.append("img2", f2);
    if(cif) fd.append("cif", cif);

    let res = await fetch("/chat", {method:"POST", body:fd});
    let data = await res.json();
    chat.innerHTML += `<div class="msg bot">${data.result}</div>`;
    chat.scrollTop = chat.scrollHeight;
}
</script>
</body>
</html>
"""

def save_file(upload_file: Optional[UploadFile]) -> Optional[str]:
    """保存上传文件，返回绝对路径"""
    if not upload_file:
        return None
    fname = f"{uuid.uuid4()}_{upload_file.filename}"
    save_path = os.path.join("uploads", fname)
    with open(save_path, "wb") as f:
        f.write(upload_file.file.read())
    return save_path

@app.post("/chat")
async def chat(
    msg: str = Form(...),
    img1: Optional[UploadFile] = File(None),
    img2: Optional[UploadFile] = File(None),
    cif: Optional[UploadFile] = File(None)
):
    # 保存全部文件
    p1 = save_file(img1)
    p2 = save_file(img2)
    cif_path = save_file(cif)

    try:
        # 1. 空间群预测：支持 单图 / 双图
        if "预测" in msg:
            if not p1:
                return {"result":"❌ 必须上传至少一张电镜图像"}
            # 双图自动切换多图模型
            model_type = "mvbcnn" if p2 else "svcnn"
            res = await pre_symmetry(
                image1=p1,
                image2=p2,
                model_type=model_type,
                top_k=5
            )
            out = f"✅ 【空间群预测】完成\n模式：{model_type}\n图1：{p1}\n"
            out += f"图2：{p2}\n" if p2 else ""
            out += f"\n预测结果：\n{str(res)}"
            return {"result": out}

        # 2. 双图检索
        elif "检索" in msg:
            if not p1 or not p2:
                return {"result":"❌ 图像检索必须上传【两张】SAED图像"}
            res = await retrieve_for_two_images(img1=p1, img2=p2)
            return {"result":f"✅ 【双图相似检索】完成\n结果：\n{str(res)}"}

        # 3. 实验 + CIF 模拟匹配
        elif "匹配" in msg:
            if not p1:
                return {"result":"❌ 请上传实验SAED图像"}
            res = await match_experiment_with_simulation(
                exp_img_input=p1,
                cif_input=cif_path
            )
            tip = cif_path if cif_path else "未上传CIF，使用默认晶体"
            return {"result":f"✅ 【实验-模拟图谱匹配】完成\nCIF：{tip}\n\n匹配结果：\n{str(res)}"}

        # 4. 图像入库
        elif "入库" in msg:
            img_list = []
            if p1: img_list.append(p1)
            if p2: img_list.append(p2)
            if not img_list:
                return {"result":"❌ 请上传至少一张图像"}
            add_images_to_database(images_input=img_list, sg=1, mid=1)
            return {"result":f"✅ 【图像入库】成功\n已入库文件：\n" + "\n".join(img_list)}

        else:
            info = f"✅ 消息已接收：{msg}\n"
            info += f"图像1：{p1}\n" if p1 else ""
            info += f"图像2：{p2}\n" if p2 else ""
            info += f"CIF文件：{cif_path}" if cif_path else ""
            return {"result": info}

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return {"result": f"❌ 运行异常：\n{str(e)}\n\n{err[:1500]}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)