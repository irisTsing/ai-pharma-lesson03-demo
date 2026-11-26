#!/usr/bin/env python
"""数据准备：从 DUD-E DRD3 原始文件构建筛选库、划分、对接盒子和对接子集。

运行（在 demos/demo_vs 下）：
    ../venv_chemprop/bin/python common/prepare_data.py

产出：
    data/<target>_library_10k.csv    480 actives + 9520 decoys（固定 seed=42），demo 1/2 用
    data/<target>_dock_library.csv   480 actives + 1520 decoys（从上面库中再抽），demo 3/4 用
    data/<target>_split.json         scaffold 划分与随机划分的 train/test 下标、参考活性分子下标
    data/<target>_box.json           对接盒子中心与尺寸（由 crystal_ligand.mol2 质心确定）
所有随机抽样都固定种子，重复运行得到同一份数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_common as vc


def crystal_ligand_center(mol2_path: Path) -> dict:
    """读 DUD-E 的 crystal_ligand.mol2，返回质心与包围盒范围。"""
    pts = []
    in_atoms = False
    for line in mol2_path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atoms = False
            continue
        if in_atoms and line.strip():
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                pts.append([float(parts[2]), float(parts[3]), float(parts[4])])
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 5:
        raise SystemExit(f"从 {mol2_path} 没读到原子坐标")
    center = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return {
        "center": [round(float(x), 3) for x in center],
        "crystal_ligand_extent": [round(float(x), 3) for x in extent],
        "crystal_ligand_n_atoms": int(len(pts)),
    }


def check_pocket(center: dict, receptor_pdb: Path) -> dict:
    """检查晶体配体周围是否有受体原子，确认盒子落在口袋里。"""
    lig = crystal_ligand_coords(center)  # 下面用中心代替，另算最近距离
    rec = []
    for line in receptor_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                rec.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    rec = np.asarray(rec, dtype=np.float64)
    d = np.linalg.norm(rec - np.asarray(center["center"])[None, :], axis=1)
    return {
        "receptor_atoms": int(len(rec)),
        "min_dist_center_to_receptor_A": round(float(d.min()), 3),
        "n_receptor_atoms_within_10A": int((d < 10.0).sum()),
    }


def crystal_ligand_coords(center: dict):
    return np.asarray(center["center"])[None, :]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="drd3", help="DUD-E 靶点名，默认 drd3（备选 akr1c3/egfr）")
    ap.add_argument("--n-decoys", type=int, default=9520, help="库中 decoy 数，默认 9520（库共 10000）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dock-n-decoys", type=int, default=1520, help="对接子集中 decoy 数，默认 1520（对接库共 2000）")
    ap.add_argument("--dock-seed", type=int, default=7)
    ap.add_argument("--n-refs", type=int, default=10, help="demo1 使用的参考活性分子数")
    ap.add_argument("--force", action="store_true", help="已有文件也重新生成")
    args = ap.parse_args()

    vc.set_matplotlib_cache()
    target_dir = vc.DATA_DIR / args.target
    if not (target_dir / "actives_final.ism").exists():
        raise SystemExit(
            f"找不到 {target_dir}/actives_final.ism。先运行 data/download_data.sh 下载 DRD3。"
        )

    lib_csv = vc.DATA_DIR / f"{args.target}_library_10k.csv"
    if lib_csv.exists() and not args.force:
        print(f"[skip] {lib_csv} 已存在（加 --force 重新生成）")
        lib = vc.load_library(lib_csv)
    else:
        df = vc.build_library(target_dir, n_decoys=args.n_decoys, seed=args.seed, out_csv=lib_csv)
        lib = vc.load_library(lib_csv)
        print(f"[library] {len(df)} 分子 -> {lib_csv}")
    n_act = int(lib["label"].sum())
    print(f"[library] actives={n_act} decoys={len(lib['label']) - n_act}")

    # 对接子集：全部 actives + 从库内 decoy 再抽，保证是库的子集，口径清楚
    dock_csv = vc.DATA_DIR / f"{args.target}_dock_library.csv"
    if dock_csv.exists() and not args.force:
        print(f"[skip] {dock_csv} 已存在（加 --force 重新生成）")
    else:
        import pandas as pd

        df = lib["df"]
        act = df[df["label"] == 1].reset_index(drop=True)
        dec = df[df["label"] == 0].reset_index(drop=True)
        rng = np.random.RandomState(args.dock_seed)
        dec_idx = np.sort(rng.choice(len(dec), size=min(args.dock_n_decoys, len(dec)), replace=False))
        dock = pd.concat([act, dec.iloc[dec_idx]], ignore_index=True)
        dock.to_csv(dock_csv, index=False)
        print(f"[dock_library] {len(dock)} 分子（actives={int(dock['label'].sum())}）-> {dock_csv}")

    # 划分：scaffold（主口径，避免泄漏）与随机（对照）
    split_path = vc.DATA_DIR / f"{args.target}_split.json"
    if split_path.exists() and not args.force:
        print(f"[skip] {split_path} 已存在（加 --force 重新生成）")
        splits = json.loads(split_path.read_text())
    else:
        y = lib["label"]
        smiles = lib["smiles"]
        tr_s, te_s = vc.scaffold_split(y, smiles, frac_test=0.2, seed=2024)
        tr_r, te_r = vc.random_split(y, frac_test=0.2, seed=2024)
        # 参考活性分子只从训练侧取，保证 demo1 与 demo2 用同一套已知信息
        ref_pool = [i for i in tr_s.tolist() if y[i] == 1]
        rng = np.random.RandomState(11)
        refs = sorted(rng.choice(ref_pool, size=min(args.n_refs, len(ref_pool)), replace=False).tolist())
        splits = {
            "scaffold": {
                "train": tr_s.tolist(),
                "test": te_s.tolist(),
                "summary": vc.describe_split(y, tr_s, te_s, "scaffold"),
            },
            "random": {
                "train": tr_r.tolist(),
                "test": te_r.tolist(),
                "summary": vc.describe_split(y, tr_r, te_r, "random"),
            },
            "reference_actives": refs,
        }
        vc.save_json(splits, split_path)
        for k in ("scaffold", "random"):
            print(f"[split:{k}] {splits[k]['summary']}")

    # 对接盒子
    box_path = vc.DATA_DIR / f"{args.target}_box.json"
    if box_path.exists() and not args.force:
        print(f"[skip] {box_path} 已存在（加 --force 重新生成）")
        box = json.loads(box_path.read_text())
    else:
        box = crystal_ligand_center(target_dir / "crystal_ligand.mol2")
        box.update({"size": [20.0, 20.0, 20.0], "target": args.target})
        box["pocket_check"] = check_pocket(box, target_dir / "receptor.pdb")
        vc.save_json(box, box_path)
        print(f"[box] center={box['center']} size={box['size']} -> {box_path}")
        print(f"[box] 口袋检查 {box['pocket_check']}")

    print("\n准备完成。")


if __name__ == "__main__":
    main()
