import base64
import io
import numpy as np
from PIL import Image

def base64_to_np(base64_str: str) -> np.ndarray:
    """
    base64 -> np.ndarray (RGB)
    """

    # 去 header
    if "," in base64_str and "base64" in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return np.array(img).astype(np.uint8)

import numpy as np
import torch

def make_json_safe(obj):

    if obj is None:
        return None

    # bytes → base64（避免 UTF-8 crash）
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")

    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    # torch
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    # dict
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    return obj

import os
import numpy as np
import json

# =====================================================
# 安全输出（必须强化版）
# =====================================================
def safe_output(obj):

    if obj is None:
        return None

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("utf-8")

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): safe_output(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [safe_output(v) for v in obj]

    return str(obj)