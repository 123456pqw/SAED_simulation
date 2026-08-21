import sys
from pathlib import Path
import argparse

import os
import numpy as np
from PIL import Image
import torch
import sys
import os
sys.path.append('/share/STEMCrysNet/')
from eval_utils import get_eval_gradio, construct_input, get_model_gradio
from typing import Optional
from mcp.server.fastmcp import FastMCP

HYDRA_FULL_ERROR=1
def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1.0e-10)

# def parse_args():
#     """Parse command line arguments for MCP server."""
#     parser = argparse.ArgumentParser(description="STEM to Crystal MCP Server")
#     parser.add_argument('--port', type=int, default=50002, help='Server port (default: 50002)')
#     parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
#     parser.add_argument('--log-level', default='INFO', 
#                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
#                        help='Logging level (default: INFO)')
#     try:
#         args = parser.parse_args()
#     except SystemExit:
#         class Args:
#             port = 50002
#             host = '0.0.0.0'
#             log_level = 'INFO'
#         args = Args()
#     return args

# args = parse_args()
# mcp = FastMCP("stem_to_crystal", port=args.port, host=args.host)


def get_stem_2_crystal(formula: str, eval_num: int, file_path: str, pixel_size: str, save_dir: str, file_path2: Optional[str] = None, pixel_size2: Optional[str] = None) -> str:
    """
    Convert STEM image to crystal structure.

    Args:
        formula: Chemical formula
        eval_num: Number of evaluations
        file_path: Path to the STEM image file
        pixel_size: Pixel size of the first STEM image (nm/pixel)
        save_dir: Directory to save results
        file_path2: Optional path to the second STEM image file
        pixel_size2: Optional pixel size of the second STEM image (nm/pixel), defaults to pixel_size if not provided

    Returns:
        Path to the output file
    """
    if file_path2 is None:
        ckpt = "/share/STEMCrysNet/model_ckpt/one_view/epoch=334-step=848555_cleaned.ckpt"
    else:
        ckpt = "/share/STEMCrysNet/model_ckpt/two_view/epoch=159-step=405440_cleaned.ckpt"
    uni3_model = get_model_gradio(ckpt)
    ccsg_model = uni3_model.cross_diff_module 
    save_path = f'{save_dir}/{formula}'
    if torch.cuda.is_available():
        ccsg_model.to("cuda")
    cpcp_model = uni3_model.contrastive_module
    os.makedirs(save_path, exist_ok=True)
    
    # Load and process first image
    pixel_size = float(pixel_size)
    stem_img_pil = Image.open(file_path).convert("L")
    w, h = stem_img_pil.size
    factor = pixel_size / 0.015
    new_width = int(w / factor)
    new_height = int(h / factor)
    stem_img_pil = stem_img_pil.resize((new_width, new_height), Image.LANCZOS)
    stem_img_pil = stem_img_pil.crop((0, 0, 256, 256))
    stem_img = np.array(stem_img_pil)
    stem_img = normalize(stem_img)
    
    # Load and process second image if provided
    if file_path2:
        pixel_size_2 = float(pixel_size2) if pixel_size2 is not None else pixel_size
        stem_img2_pil = Image.open(file_path2).convert("L")
        w, h = stem_img2_pil.size
        factor2 = pixel_size_2 / 0.015
        new_width2 = int(w / factor2)
        new_height2 = int(h / factor2)
        stem_img2_pil = stem_img2_pil.resize((new_width2, new_height2), Image.LANCZOS)
        stem_img2_pil = stem_img2_pil.crop((0, 0, 256, 256))
        stem_img2 = np.array(stem_img2_pil)
        stem_img2 = normalize(stem_img2)
    else:
        stem_img2 = stem_img
    input_data = construct_input(formula, stem_img, stem_img2)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    out_path = get_eval_gradio(
        input_data, ccsg_model, cpcp_model, save_path, num_evals=int(eval_num), model='flow'
    )
    return out_path

if __name__ == "__main__":
    input_data = {
        "formula": "Cr6Br18",
        "eval_num": 1,
        "file_path": "/share/STEMCrysNet/CrBr3_0006_ps0.015nm(1).tif",
        "pixel_size": "0.015",
        "save_dir": "/share/STEMCrysNet/",
        }
    get_stem_2_crystal(**input_data)


