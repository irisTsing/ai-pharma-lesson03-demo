#!/usr/bin/env python
"""demo 3：基于结构的药物筛选——GPU 批量对接（Uni-Dock）与 CPU 对照（vina）。

流程：
1. 受体准备：DUD-E 的 receptor.pdb -> obabel -xr -p 7.4 -> receptor.pdbqt；
2. 配体准备：DUD-E 附带 3D mol2 -> obabel -p 7.4 -> pdbqt；
3. GPU：一次调用把 10 个配体交给 Uni-Dock（--gpu_batch，fast 模式）；
4. CPU 对照：同一批里取前 5 个配体，用 AutoDock Vina --cpu 2 跑同样的盒子；
5. 输出打分表、每个配体的最优 pose、单配体耗时与加速比。

GPU 不可用时（比如在开发机上）自动只跑 vina CPU 版本，并明确提示。
本机（无 NVIDIA 驱动）验证了 CPU 路径；GPU 路径待 gpu1 首次验证（清单见 README）。

运行：
    ../venv_chemprop/bin/python demo_docking/run_docking.py --device auto
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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

OBABEL = "/usr/local/bin/obabel"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--ligands", default=str(vc.DATA_DIR / "demo3_ligands.csv"))
    ap.add_argument("--n-ligands", type=int, default=10)
    ap.add_argument("--vina-n", type=int, default=5, help="CPU 对照的配体数")
    ap.add_argument("--vina-cpu", type=int, default=2, help="vina 线程数；课堂每人 2～4")
    ap.add_argument("--vina-exhaustiveness", type=int, default=4, help="课堂预算 4，默认 8 更慢")
    ap.add_argument("--vina-workers", type=int, default=2)
    ap.add_argument("--unidock-mode", default="fast", choices=["fast", "balance", "detail"])
    ap.add_argument("--unidock-mem-mb", type=int, default=2500, help="Uni-Dock 显存上限（MB）；50 人共用一张卡必须设")
    ap.add_argument("--out-dir", default=str(DEMO_DIR / "results"))
    return ap.parse_args()


def prepare_receptor() -> Path:
    pdbqt = vc.DATA_DIR / vc.TARGET / "receptor.pdbqt"
    t0 = time.perf_counter()
    if not pdbqt.exists() or (vc.DATA_DIR / vc.TARGET / "receptor.pdb").stat().st_mtime > pdbqt.stat().st_mtime:
        proc = subprocess.run(
            ["bash", str(DEMO_VS_DIR / "common/prepare_receptor.sh"), vc.TARGET],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"受体准备失败：\n{proc.stdout}\n{proc.stderr}")
        print(proc.stdout.strip())
    else:
        print(f"[receptor] 复用 {pdbqt}（FORCE=1 可重建）")
    print(f"[receptor] 耗时 {time.perf_counter() - t0:.2f} s")
    return pdbqt


def prepare_ligands(df: pd.DataFrame, out_dir: Path) -> tuple[list[Path], float]:
    """从 DUD-E 的 3D mol2 用 obabel 转 pdbqt。学生现场看到的就是这一步。"""
    mol2_dir = vc.DATA_DIR / "dock_ligands/mol2"
    pdbqt_dir = out_dir / "ligands_pdbqt"
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    t0 = time.perf_counter()
    for mol_id in df["mol_id"]:
        src = mol2_dir / f"{mol_id}.mol2"
        dst = pdbqt_dir / f"{mol_id}.pdbqt"
        if not dst.exists():
            proc = subprocess.run([OBABEL, str(src), "-O", str(dst), "-p", "7.4"], capture_output=True, text=True)
            if not dst.exists():
                raise SystemExit(f"配体 {mol_id} 转 pdbqt 失败：{proc.stderr[-300:]}")
        paths.append(dst)
    elapsed = time.perf_counter() - t0
    print(f"[ligands] {len(paths)} 个配体 -> pdbqt，耗时 {elapsed:.2f} s（obabel -p 7.4）")
    return paths, elapsed


def export_best_pose(src: Path | None, mol_id: str, pose_dir: Path) -> str:
    """把最优 pose 复制到 results/poses/，vina 的 pdbqt 另存一份 pdb 方便可视化。"""
    if src is None or not src.exists():
        return ""
    pose_dir.mkdir(parents=True, exist_ok=True)
    dst = pose_dir / f"{mol_id}_best{src.suffix}"
    shutil.copyfile(src, dst)
    if src.suffix == ".pdbqt":
        pdb = pose_dir / f"{mol_id}_best.pdb"
        subprocess.run([OBABEL, str(dst), "-O", str(pdb)], capture_output=True, text=True)
    return str(dst.relative_to(DEMO_VS_DIR))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    print("=" * 70)
    print("demo 3：基于结构的筛选（Uni-Dock GPU / AutoDock Vina CPU）")
    print("=" * 70)
    device = vc.pick_device(args.device)
    env = vc.env_report(device)
    print(f"[env] device={device}  torch {env.get('torch')}  rdkit {env.get('rdkit')}")

    df = pd.read_csv(args.ligands).head(args.n_ligands).reset_index(drop=True)
    print(f"[data] 配体 {len(df)} 个（活性 {int(df['label'].sum())}，诱饵 {int((df['label'] == 0).sum())}）")

    receptor = prepare_receptor()
    ligand_paths, lig_prep_s = prepare_ligands(df, out_dir)
    box = vc.load_box(vc.DATA_DIR / "drd3_box.json")
    print(f"[box] center={box['center']}  size={box['size']}（由晶体配体质心确定）")

    unidock_df = None
    gpu_status = "skipped"
    if device == "cuda":
        try:
            unidock_df = dr.run_unidock(
                ligand_paths,
                receptor,
                box,
                out_dir / "unidock_poses",
                search_mode=args.unidock_mode,
                scoring="vina",
                num_modes=1,
                seed=2024,
                max_gpu_memory_mb=args.unidock_mem_mb,
            )
            gpu_status = "ok"
            u = unidock_df[unidock_df["status"] == "ok"]
            print(
                f"[unidock] {len(u)}/{len(unidock_df)} 个配体成功，"
                f"总耗时 {unidock_df['wall_total_s'].iloc[0]:.2f} s，"
                f"平均 {unidock_df['wall_per_lig_s'].iloc[0]:.2f} s/配体（fast 模式）"
            )
        except Exception as exc:
            gpu_status = "failed"
            print(f"[unidock] GPU 路径失败，回退 CPU vina：{exc}")
    else:
        print("[unidock] --device cpu，跳过 GPU 批量对接")

    # CPU 对照：vina 跑前 5 个配体
    n_vina = min(args.vina_n, len(ligand_paths))
    vina_ligands = ligand_paths[:n_vina]
    print(f"\n[vina] CPU 对照：{n_vina} 个配体，--cpu {args.vina_cpu}，exhaustiveness {args.vina_exhaustiveness}")
    vina_df = dr.run_vina_parallel(
        vina_ligands,
        receptor,
        box,
        out_dir / "vina_poses",
        cpu=args.vina_cpu,
        exhaustiveness=args.vina_exhaustiveness,
        num_modes=1,
        workers=args.vina_workers,
        seed=2024,
        keep_poses=True,
        tag="vina",
    )
    v = vina_df[vina_df["status"] == "ok"]
    vina_total_s = float(vina_df["wall_total_s"].iloc[0])
    vina_mean_s = float(v["wall_s"].mean())
    print(
        f"[vina] {len(v)}/{len(vina_df)} 成功，总耗时 {vina_total_s:.1f} s，"
        f"平均 {vina_mean_s:.1f} s/配体"
    )

    # 若没有 GPU，则用 vina 把 10 个配体都跑一遍，保证 demo 3 在纯 CPU 上也有完整结果
    if unidock_df is None:
        rest = ligand_paths[n_vina:]
        if rest:
            print(f"[vina] 无 GPU，继续用 vina 跑剩余 {len(rest)} 个配体")
            rest_df = dr.run_vina_parallel(
                rest, receptor, box, out_dir / "vina_poses", cpu=args.vina_cpu,
                exhaustiveness=args.vina_exhaustiveness, num_modes=1,
                workers=args.vina_workers, seed=2024, keep_poses=True, tag="vina",
            )
            vina_df = pd.concat([vina_df, rest_df], ignore_index=True)
        unidock_df = vina_df.copy()
        unidock_df["method"] = "vina_cpu_fallback"
        if "wall_per_lig_s" not in unidock_df:
            unidock_df["wall_per_lig_s"] = unidock_df["wall_s"]

    # 打分表 + 对照表
    label_map = dict(zip(df["mol_id"], df["label"]))
    scores = unidock_df.copy()
    scores["label"] = scores["ligand"].map(label_map)
    scores = scores.rename(columns={"ligand": "mol_id"})
    keep = ["mol_id", "label", "method", "status", "affinity", "wall_s", "wall_total_s", "wall_per_lig_s"]
    for c in keep:
        if c not in scores:
            scores[c] = np.nan
    scores = scores[keep]
    scores["invalid_positive_score"] = scores["affinity"] > 0
    scores.to_csv(out_dir / "docking_scores.csv", index=False)

    if gpu_status == "ok":
        compare = dr.summarize_compare(vina_df, unidock_df)
        compare["label"] = compare["ligand"].map(label_map)
    else:
        # 没有 GPU 时不编造 GPU 列，只保留 vina 对照，避免把回退结果当成 GPU 结果
        compare = vina_df[["ligand", "status", "affinity"]].rename(
            columns={"status": "vina_status", "affinity": "vina_affinity"}
        )
        compare["unidock_status"] = "skipped_no_gpu"
        compare["unidock_affinity"] = np.nan
        compare["delta_kcal"] = np.nan
        compare["label"] = compare["ligand"].map(label_map)
    compare.to_csv(out_dir / "compare_cpu_gpu.csv", index=False)

    # pose 导出
    pose_dir = out_dir / "poses"
    for mol_id in df["mol_id"]:
        if unidock_df is not None and unidock_df["method"].iloc[0].startswith("unidock"):
            udoc = out_dir / "unidock_poses"
            src = None
            for cand in (udoc / f"{mol_id}_out.pdbqt", udoc / f"{mol_id}_out.sdf"):
                if cand.exists():
                    src = cand
                    break
            export_best_pose(src, mol_id, pose_dir)
        else:
            export_best_pose(out_dir / "vina_poses/poses" / f"{mol_id}.pdbqt", mol_id, pose_dir)

    timing = {
        "device": device,
        "gpu_status": gpu_status,
        "receptor_prep_s": None,
        "ligand_prep_s": round(lig_prep_s, 2),
        "vina": {
            "n": int(len(vina_df)),
            "cpu": args.vina_cpu,
            "exhaustiveness": args.vina_exhaustiveness,
            "total_wall_s": vina_total_s,
            "mean_s_per_ligand": round(vina_mean_s, 2),
        },
        "unidock": None,
        "total_s": None,
    }
    if gpu_status == "ok":
        uok = unidock_df[unidock_df["status"] == "ok"]
        timing["unidock"] = {
            "search_mode": args.unidock_mode,
            "n": int(len(unidock_df)),
            "total_wall_s": float(unidock_df["wall_total_s"].iloc[0]),
            "mean_s_per_ligand": float(unidock_df["wall_per_lig_s"].iloc[0]),
            "gpu_mem_limit_mb": args.unidock_mem_mb,
            "speedup_vs_vina": round(
                timing["vina"]["mean_s_per_ligand"] / max(1e-9, float(unidock_df["wall_per_lig_s"].iloc[0])), 2
            ),
        }
    timing["total_s"] = round(time.perf_counter() - t_all, 2)
    vc.save_json(timing, out_dir / "timing.json")

    # 终端汇总
    print("\n[打分表]")
    print(scores.drop(columns=["wall_total_s", "wall_per_lig_s"]).to_string(index=False))
    if gpu_status == "ok":
        print("\n[CPU vs GPU 对照]")
        print(compare[["ligand", "label", "vina_affinity", "unidock_affinity", "delta_kcal"]].to_string(index=False))
    act = scores[scores["label"] == 1]["affinity"].mean()
    dec = scores[scores["label"] == 0]["affinity"].mean()
    print(f"\n[活性 vs 诱饵] 平均打分 活性 {act:.2f} kcal/mol，诱饵 {dec:.2f} kcal/mol")
    print(f"[timing] {json.dumps(timing, ensure_ascii=False)}")
    print(f"\n[output] {out_dir}/docking_scores.csv")
    print(f"[output] {out_dir}/compare_cpu_gpu.csv")
    print(f"[output] {out_dir}/timing.json")
    print(f"[output] {pose_dir}/ 每个配体的最优 pose")
    print(f"[total] 全流程 {vc.fmt_secs(timing['total_s'])}")


if __name__ == "__main__":
    main()
