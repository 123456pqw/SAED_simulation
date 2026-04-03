import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from .run_saed import run_ed_simulation  
import json

def save_summary(spacegroup_map, output_csv):
    """
    保存空间群材料数量统计到CSV文件。
    """
    summary = []
    for sg, mids in spacegroup_map.items():
        summary.append({'spacegroup': sg, 'count': len(mids)})
    df = pd.DataFrame(summary)
    df.to_csv(output_csv, index=False)
    print(f"✅ 空间群统计已保存到 {output_csv}")

def extract_spacegroup_number(spacegroup_str):
    try:
        if pd.isna(spacegroup_str) or str(spacegroup_str).strip() in ('', 'None', '{}'):
            return None
        json_str = str(spacegroup_str).replace("'", '"')
        match = re.search(r'"number"\s*:\s*(\d+)', json_str)
        if match:
            number = match.group(1)
        match = re.search(r'number[&#39;"]?\s*:\s*(\d+)', json_str)
        if match:
            number = match.group(1)
        return number
    except:
        return None


def load_materials_by_spacegroup(csv_path, source_dir):
    df = pd.read_csv(csv_path, sep=';')
    spacegroup_to_materials = defaultdict(list)
    source_materials = {
        d[:-4] for d in os.listdir(source_dir)
        if d.startswith('mp-') or d.startswith('mvc-')
    }
    #print(source_materials)
    #print(len(source_materials))
    count=0
    for _, row in df.iterrows():
        mid = row['material_id']
        if mid not in source_materials:
            continue
        count += 1
        sg = extract_spacegroup_number(row['spacegroup'])
        if sg is not None:
            spacegroup_to_materials[sg].append(mid)
    #print(f"✅ 共找到 {count} 个材料的空间群信息")
    OUTPUT_CSV = "spacegroup_count_summary.csv"
    #save_summary(spacegroup_to_materials, OUTPUT_CSV)
    return spacegroup_to_materials


def generate_beam_directions():
    directions = [[0, 0, 1]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == j == k == 0:
                    continue
                vec = np.array([i, j, k])
                angles = [
                    np.arccos(np.clip(np.dot(vec/np.linalg.norm(vec), np.array(d)/np.linalg.norm(d)), -1.0, 1.0))
                    for d in directions
                ]
                if np.min(angles) > 1e-3:
                    directions.append([i, j, k])
    return directions


def simulate_one_material(args):
    sg, mid, root_dir, save_root, beam_list = args
    cif_path = os.path.join(root_dir, f"{mid}.cif")
    if not os.path.isfile(cif_path):
        print(f"[跳过] {mid} CIF文件不存在")
        return

    out_dir = os.path.join(save_root, str(sg), mid)
    os.makedirs(out_dir, exist_ok=True)

    for beam in beam_list:
        beam_tag = f"beam_{beam[0]}_{beam[1]}_{beam[2]}"
        out_path = os.path.join(out_dir, f"{beam_tag}.png")

        if os.path.isfile(out_path):
            continue

        try:
            run_ed_simulation(cif_path, zone_axis=beam, filename=out_path)
        except Exception as e:
            print(f"[失败] {mid} @ {beam_tag}，错误：{e}")

'''
def main():
    ROOT_DIR = "/internfs/pengqianwen/stem-crystal-system/single_opinion/Data/MP_cifs_processed"
    SAVE_ROOT = "/internfs/pengqianwen/MVBCNN/data"
    CSV_PATH = "/internfs/pengqianwen/stem-crystal-system/single_opinion/Data/file_id.csv"

    beam_direction_round = generate_beam_directions()
    spacegroup_map = load_materials_by_spacegroup(CSV_PATH, ROOT_DIR)

    all_args = []
    for sg, mids in spacegroup_map.items():
        for mid in mids:
            all_args.append((sg, mid, ROOT_DIR, SAVE_ROOT, beam_direction_round))

    print(f"共计任务数: {len(all_args)}")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    pool = Pool(processes=8)
    for _ in tqdm(pool.imap_unordered(simulate_one_material, all_args), total=len(all_args)):
        pass
    pool.close()
    pool.join()
'''
def main():
    import sys
    if len(sys.argv) < 2:
        print("❌ 用法: python run_multem_batch.py task_chunk_X.json")
        return

    task_file = sys.argv[1]
    with open(task_file, "r") as f:
        all_args = json.load(f)

    print(f"🚀 加载任务文件 {task_file}，任务数: {len(all_args)}")
    pool = Pool(processes=8)
    for _ in tqdm(pool.imap_unordered(simulate_one_material, all_args), total=len(all_args)):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
