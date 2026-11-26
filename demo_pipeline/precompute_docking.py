#!/usr/bin/env python
"""demo 4 第一步（教师课前跑）：对对接子集的全部配体做 vina 打分，结果存 CSV。

本机（64 核，无 GPU）实测：1,981 个配体、20 进程 × 每进程 3 线程、exhaustiveness=4，
约 30 分钟完成。GPU 节点上可以用 demo_docking 的 Uni-Dock 路径替代，速度快一到两个数量级。

支持断点续跑：已经在 CSV 里的配体直接跳过。
打分异常的配体（affinity > 0，通常是配体 pdbqt 的扭转树有问题）标为 invalid，
不参与富集分析，数量与原因写进 meta json。

用法（在 demos/demo_vs 下）：
    ../venv_chemprop/bin/python demo_pipeline/precompute_docking.py \
        --workers 20 --cpu 3 --out demo_pipeline/results/full_docking.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
DEMO_VS_DIR = DEMO_DIR.parent
sys.path.insert(0, str(DEMO_VS_DIR / "common"))
import dock_run as dr  # noqa: E402
import vs_common as vc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(vc.DATA_DIR / "drd3_dock_library.csv"))
    ap.add_argument("--ligand-dir", default=str(vc.DATA_DIR / "dock_ligands/pdbqt"))
    ap.add_argument("--prep-summary", default=str(vc.DATA_DIR / "dock_ligands/prep_summary.csv"))
    ap.add_argument("--receptor", default=str(vc.DATA_DIR / "drd3/receptor.pdbqt"))
    ap.add_argument("--box", default=str(vc.DATA_DIR / "drd3_box.json"))
    ap.add_argument("--out", default=str(DEMO_DIR / "results/full_docking.csv"))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cpu", type=int, default=3, help="每个 vina 进程的线程数")
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max-ligands", type=int, default=0)
    ap.add_argument("--keep-poses", action="store_true")
    ap.add_argument("--chunk", type=int, default=150, help="每跑完多少配体落盘一次，便于随时取部分结果")
    ap.add_argument("--method", default="vina", choices=["vina", "unidock"],
                    help="vina=CPU 逐配体（本机约 2 小时）；unidock=GPU 批量（gpu1，快一到两个数量级）")
    ap.add_argument("--search-mode", default="fast", choices=["fast", "balance", "detail"])
    ap.add_argument("--unidock-mem-mb", type=int, default=3000, help="Uni-Dock 显存上限（MB）")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    box = vc.load_box(args.box)
    lib = pd.read_csv(args.library)
    prep = pd.read_csv(args.prep_summary)
    ok_ids = set(prep.loc[prep["status"] == "ok", "mol_id"].astype(str))
    lib = lib[lib["mol_id"].astype(str).isin(ok_ids)].reset_index(drop=True)
    # 打乱配体顺序：这样预计算跑了一部分时，已完成的部分里活性与诱饵比例接近全库，
    # 学生不用等全部跑完就能做 demo 4 的分析。
    lib = lib.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_ligands:
        lib = lib.head(args.max_ligands)

    done = set()
    if out.exists():
        old = pd.read_csv(out)
        # 兼容两种列名：precompute 自己写的是 mol_id，collect_dock_logs.py 也写 mol_id
        key_col = "mol_id" if "mol_id" in old.columns else ("ligand" if "ligand" in old.columns else None)
        if key_col is not None:
            done = set(old[key_col].astype(str))
        print(f"[resume] 已有 {len(done)} 个配体结果，跳过")
    todo = lib[~lib["mol_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"[precompute] 待跑 {len(todo)}/{len(lib)} 个配体，{args.workers} 进程 × {args.cpu} 线程")


    # ------------------------------------------------------------------
    # GPU 预计算路径（gpu1 推荐）：Uni-Dock 批量对接，再写同一张 CSV
    # ------------------------------------------------------------------
    if args.method == "unidock":
        ok, msg = dr.gpu_selfcheck()
        if not ok:
            raise SystemExit(f"Uni-Dock 需要可用 GPU：{msg}\n先用 vina CPU 路径，或到 gpu1 上执行本脚本。")
        print(f"[unidock] GPU {msg}；search_mode={args.search_mode}，显存上限 {args.unidock_mem_mb} MB")
        t_start = time.perf_counter()
        out_dir = out.parent
        merged = pd.read_csv(out) if out.exists() else pd.DataFrame()
        done_ids = set(merged["mol_id"].astype(str)) if len(merged) else set()
        todo = todo[~todo["mol_id"].astype(str).isin(done_ids)].reset_index(drop=True)
        batch_size = max(1, args.chunk)
        for b0 in range(0, len(todo), batch_size):
            part = todo.iloc[b0 : b0 + batch_size]
            paths = [Path(args.ligand_dir) / f"{m}.pdbqt" for m in part["mol_id"]]
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            df = dr.run_unidock(
                paths,
                args.receptor,
                box,
                out_dir / "unidock_out",
                search_mode=args.search_mode,
                scoring="vina",
                num_modes=1,
                seed=args.seed,
                max_gpu_memory_mb=args.unidock_mem_mb,
            )
            df = df.rename(columns={"ligand": "mol_id"})
            df["valid"] = df["affinity"].notna() & (df["affinity"] < 0)
            df["status"] = np.where(df["valid"], "ok", "invalid_positive_score")
            df["method"] = f"unidock_{args.search_mode}"
            df["wall_s"] = df["wall_per_lig_s"]
            df = df.merge(lib[["mol_id", "label"]], on="mol_id", how="left", suffixes=("", "_lib"))
            keep = ["mol_id", "label", "affinity", "valid", "status", "method", "wall_s", "wall_total_s", "wall_per_lig_s"]
            df = df[[c for c in keep if c in df.columns]]
            merged = pd.concat([merged, df], ignore_index=True) if len(merged) else df
            merged = merged.drop_duplicates(subset=["mol_id"], keep="last")
            merged.to_csv(out, index=False)
            print(
                f"[unidock] 本批 {len(df)} 个，累计 {len(merged)}/{len(lib)}，"
                f"总耗时 {(time.perf_counter() - t_start) / 60:.1f} 分钟 -> {out}",
                flush=True,
            )
        print(f"[unidock] 完成，用时 {(time.perf_counter() - t_start) / 60:.1f} 分钟")
        vc.save_json(
            {
                "method": f"unidock_{args.search_mode}",
                "n_docked": int(len(merged)),
                "n_valid": int(merged["valid"].sum()),
                "batch_size": batch_size,
                "gpu_mem_limit_mb": args.unidock_mem_mb,
                "box": box,
                "protocol": "Uni-Dock v1.2.0 / vina scoring / num_modes=1 / seed 固定",
            },
            out.parent / "full_docking_meta.json",
        )
        return

    t_start = time.perf_counter()
    written = 0
    if len(todo):
        for c0 in range(0, len(todo), args.chunk):
            part = todo.iloc[c0 : c0 + args.chunk]
            paths = [Path(args.ligand_dir) / f"{m}.pdbqt" for m in part["mol_id"]]
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            t0 = time.perf_counter()
            df = dr.run_vina_parallel(
                paths,
                args.receptor,
                box,
                DEMO_DIR / "results/dock_work",
                cpu=args.cpu,
                exhaustiveness=args.exhaustiveness,
                num_modes=1,
                workers=args.workers,
                seed=args.seed,
                keep_poses=args.keep_poses,
                tag="vina",
            )
            df = df.rename(columns={"ligand": "mol_id"})
            # 打分异常：affinity > 0 说明 pose 严重冲突，视为准备失败
            df["valid"] = df["affinity"].notna() & (df["affinity"] < 0)
            merged = lib.merge(df, on="mol_id", how="inner")
            if out.exists():
                old = pd.read_csv(out)
                merged = pd.concat([old, merged], ignore_index=True)
                merged = merged.drop_duplicates(subset=["mol_id"], keep="last")
            merged.to_csv(out, index=False)
            written += len(df)
            print(
                f"[precompute] 本块 {len(df)} 个，累计 {written}/{len(todo)}，"
                f"用时 {(time.perf_counter() - t0) / 60:.1f} 分钟，总进度 {len(merged)}/{len(lib)} -> {out}",
                flush=True,
            )
        elapsed = time.perf_counter() - t_start
        print(f"[precompute] 全部完成，用时 {elapsed / 60:.1f} 分钟")
    else:
        print("[precompute] 全部已完成")

    final = pd.read_csv(out)
    n_ok = int(final["valid"].sum()) if "valid" in final else 0
    meta = {
        "n_library": len(lib),
        "n_docked": int(len(final)),
        "n_valid": n_ok,
        "n_invalid_positive_score": int((~final["valid"]).sum()) if "valid" in final else 0,
        "workers": args.workers,
        "cpu_per_worker": args.cpu,
        "exhaustiveness": args.exhaustiveness,
        "box": box,
        "protocol": "vina 1.2.3, scoring=vina, num_modes=1, seed 固定",
    }
    vc.save_json(meta, out.parent / "full_docking_meta.json")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
