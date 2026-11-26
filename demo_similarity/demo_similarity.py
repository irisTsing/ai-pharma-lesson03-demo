#!/usr/bin/env python
"""demo 1：基于 2D 指纹的相似性检索与虚拟筛选评价。

方法：ECFP4（Morgan radius=2，2048 bit）+ Tanimoto 相似度。
把 10 个已知活性分子作为参考（只在训练侧选，不碰测试集），
库中每个分子的打分为“与参考分子集合的最大 Tanimoto 相似度”。

评价：scaffold 划分的测试集上算 EF1%、EF5%、ROC-AUC、top-k 命中数，
与随机排序的期望对比；同时输出全库 top-hits 与富集曲线。

GPU 版本用 torch 在显存里做分块矩阵乘计算同样的 Tanimoto 矩阵，
与 CPU（numpy/BLAS）版本对比耗时，用于说明 GPU 在这类任务上到底快多少。

运行：
    ../venv_chemprop/bin/python demo_similarity/demo_similarity.py --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
DEMO_VS_DIR = DEMO_DIR.parent
sys.path.insert(0, str(DEMO_VS_DIR / "common"))
import vs_common as vc  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--library", default=str(vc.DATA_DIR / "drd3_library_10k.csv"))
    ap.add_argument("--split", default=str(vc.DATA_DIR / "drd3_split.json"))
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--radius", type=int, default=2, help="Morgan 半径，2 即 ECFP4")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--torch-threads", type=int, default=4, help="torch CPU 线程数（GPU 时不起作用）")
    ap.add_argument("--out-dir", default=str(DEMO_DIR / "results"))
    return ap.parse_args()


def timing_device_score(fp, ref_idx, device, chunk):
    """跑 CPU 与 GPU 两条路径，返回 (score, 耗时, 显存信息)。score 是每分子最大相似度。"""
    timings = {}

    t0 = time.perf_counter()
    sim_cpu = vc.tanimoto_cpu(fp, fp[ref_idx], chunk=chunk)
    timings["cpu_full_matrix_s"] = time.perf_counter() - t0
    score_cpu = sim_cpu.max(axis=1)

    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        sim_gpu = vc.tanimoto_gpu(fp, fp[ref_idx], chunk=chunk, device="cuda")
        torch.cuda.synchronize()
        timings["gpu_full_matrix_s"] = time.perf_counter() - t0
        timings["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        timings["fp_transfer_mb"] = fp.nbytes / 1024**2
        score_gpu = sim_gpu.max(axis=1)
        diff = float(np.abs(score_cpu - score_gpu).max())
        timings["max_abs_diff_cpu_vs_gpu"] = diff
        timings["speedup_cpu_over_gpu"] = timings["cpu_full_matrix_s"] / timings["gpu_full_matrix_s"]
        score = score_gpu
    else:
        score = score_cpu
    return score, timings


def main() -> None:
    args = parse_args()
    vc.set_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = vc.pick_device(args.device, require_gpu=args.require_gpu)
    if device == "cpu":
        import torch

        torch.set_num_threads(args.torch_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    print("=" * 70)
    print("demo 1：ECFP4 + Tanimoto 相似性筛选")
    print("=" * 70)
    env = vc.env_report(device)
    print(f"[env] python {env['python']}  rdkit {env.get('rdkit')}  torch {env.get('torch')}  device={device}")
    if device == "cuda":
        print(f"[env] GPU {env.get('gpu_name')}  capability {env.get('gpu_capability')}  显存 {env.get('gpu_total_mem_gb')} GB")

    lib = vc.load_library(args.library)
    splits = json.loads(Path(args.split).read_text())
    refs = splits["reference_actives"]
    test_idx = np.array(splits["scaffold"]["test"])
    y = lib["label"]
    smiles = lib["smiles"]
    print(f"[data] 库 {len(y)} 分子，actives {int(y.sum())}，decoys {len(y) - int(y.sum())}")
    print(f"[data] 参考活性分子 {len(refs)} 个（取自训练侧，测试集完全不可见）")

    # 1) 指纹
    t0 = time.perf_counter()
    fp = vc.morgan_fp_matrix(smiles, radius=args.radius, n_bits=args.n_bits)
    fp_time = time.perf_counter() - t0

    # 2) Tanimoto
    score, timings = timing_device_score(fp, refs, device, chunk=2048)
    print(f"[score] 每分子取与 {len(refs)} 个参考分子的最大 Tanimoto")
    print(f"[time] 指纹 {fp_time:.2f} s；Tanimoto CPU {timings['cpu_full_matrix_s']:.3f} s", end="")
    if device == "cuda":
        print(
            f"；GPU {timings['gpu_full_matrix_s']:.3f} s"
            f"（峰值显存 {timings['gpu_peak_mem_gb']:.3f} GB，"
            f"CPU/GPU 加速比 {timings['speedup_cpu_over_gpu']:.1f}×，"
            f"两边最大误差 {timings['max_abs_diff_cpu_vs_gpu']:.2e}）"
        )
    else:
        print("；GPU 未启用")

    # 3) 测试集口径（与 demo 2 同一批分子，便于直接对照）
    y_test = y[test_idx]
    s_test = score[test_idx]
    metrics_test = vc.summarize_scores(y_test, s_test, topk=(10, 20, 50))
    metrics_test["baseline_active_rate"] = float(y_test.mean())
    print("\n[测试集：scaffold 划分的 2000 个分子]")
    print(f"  阳性 {int(y_test.sum())}/{len(y_test)}（基准率 {y_test.mean():.1%}）")
    print(f"  ROC-AUC {metrics_test['roc_auc']:.3f}  EF1% {metrics_test['ef1']:.2f}  EF5% {metrics_test['ef5']:.2f}  EF10% {metrics_test['ef10']:.2f}")
    print(f"  top-20 命中 {metrics_test['topk_hits']['top20']}/20 个活性分子，随机期望 {20 * y_test.mean():.2f}")

    # 4) 全库口径
    metrics_full = vc.summarize_scores(y, score, topk=(10, 20, 50))
    metrics_full["baseline_active_rate"] = float(y.mean())
    print("\n[全库：10000 个分子]")
    print(f"  ROC-AUC {metrics_full['roc_auc']:.3f}  EF1% {metrics_full['ef1']:.2f}  EF5% {metrics_full['ef5']:.2f}")

    # 5) top-k 检索结果
    order = np.argsort(-score, kind="stable")
    top = order[: args.topk]
    rows = [
        {
            "rank": r + 1,
            "mol_id": lib["mol_id"][i],
            "label": "active" if y[i] == 1 else "decoy",
            "tanimoto_max": round(float(score[i]), 4),
            "smiles": smiles[i],
        }
        for r, i in enumerate(top)
    ]
    import pandas as pd

    top_df = pd.DataFrame(rows)
    top_df.to_csv(out_dir / "top_hits.csv", index=False)
    print(f"\n[top-{args.topk}] 前 10 个：")
    print(top_df.head(10).drop(columns=["smiles"]).to_string(index=False))

    # 6) 分数全表（demo 4 的粗筛直接复用，不用重算）
    scores_df = pd.DataFrame(
        {
            "mol_id": lib["mol_id"],
            "smiles": smiles,
            "label": y,
            "tanimoto_max_to_refs": np.round(score, 6),
        }
    )
    scores_df.to_csv(out_dir / "scores.csv", index=False)

    # 7) 富集曲线 + 分数分布
    xs, found = vc.enrichment_curve(y_test, s_test, n_points=200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(xs * 100, found * 100, lw=2, label="ECFP4 + Tanimoto")
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="random")
    ax.set_xlabel("fraction of library screened (%)")
    ax.set_ylabel("fraction of actives recovered (%)")
    ax.set_title(f"Enrichment curve (scaffold test set, n={len(y_test)})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    ax.hist(score[test_idx][y_test == 1], bins=bins, alpha=0.6, label=f"actives (n={int(y_test.sum())})", density=True)
    ax.hist(score[test_idx][y_test == 0], bins=bins, alpha=0.6, label=f"decoys (n={len(y_test) - int(y_test.sum())})", density=True)
    ax.set_xlabel("max Tanimoto similarity to reference actives")
    ax.set_ylabel("density")
    ax.set_title("Score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out_dir / "enrichment_curve.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t_all
    result = {
        "demo": "demo_similarity",
        "method": "ECFP4 (Morgan r=2) + Tanimoto max over reference actives",
        "device": device,
        "env": env,
        "params": vars(args),
        "n_library": len(y),
        "n_actives": int(y.sum()),
        "n_reference_actives": len(refs),
        "timing_s": {
            "fp_generation": round(fp_time, 3),
            **{k: round(v, 6) for k, v in timings.items()},
            "total": round(elapsed, 2),
        },
        "test_split": metrics_test,
        "full_library": metrics_full,
        "top_hits_preview": rows[:10],
    }
    vc.save_json(result, out_dir / "metrics.json")
    print(f"\n[output] {out_dir}/metrics.json")
    print(f"[output] {out_dir}/top_hits.csv")
    print(f"[output] {out_dir}/enrichment_curve.png")
    print(f"[total] 全流程 {vc.fmt_secs(elapsed)}")


if __name__ == "__main__":
    main()
