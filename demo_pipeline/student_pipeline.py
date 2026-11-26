#!/usr/bin/env python
"""demo 4：融合流水线——2D 特征粗筛 top 10%，再对接精排，与全库对接比成本与富集。

前提：教师已用 precompute_docking.py 得到对接子集的 vina 打分 CSV
（教师课前一键预计算，本机 CPU 约 30 分钟；gpu1 上可用 Uni-Dock 更快）。

学生这一步只做两件事：
1. 用 ECFP4 + Tanimoto（10 个参考活性分子）给 2D 特征粗筛打分，取 top 10%；
2. 把粗筛结果与全库对接结果对比：
   - 粗筛 top 10% 能召回多少比例的活性分子；
   - 只对接这 10% 与对接全库，top-20 / top-1% 的命中差多少；
   - 省下多少对接次数与机时。

运行（在 demos/demo_vs 下）：
    ../venv_chemprop/bin/python demo_pipeline/student_pipeline.py --top-frac 0.10
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
import vs_common as vc  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(vc.DATA_DIR / "drd3_dock_library.csv"))
    ap.add_argument("--split", default=str(vc.DATA_DIR / "drd3_split.json"))
    ap.add_argument("--docking-csv", default=str(DEMO_DIR / "results/full_docking.csv"))
    ap.add_argument("--top-frac", type=float, default=0.10, help="粗筛保留比例，默认前 10%")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out-dir", default=str(DEMO_DIR / "results"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    vc.set_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("=" * 70)
    print("demo 4：2D 粗筛 → 对接精排的融合流水线")
    print("=" * 70)

    dock = pd.read_csv(args.docking_csv)
    if "valid" not in dock:
        dock["valid"] = dock["affinity"].notna() & (dock["affinity"] < 0)
    dock = dock[dock["valid"]].copy()
    lib = pd.read_csv(args.library)
    lib["mol_id"] = lib["mol_id"].astype(str)
    dock["mol_id"] = dock["mol_id"].astype(str)
    # 统一用对接子集作为“全库”口径
    lib = lib[lib["mol_id"].isin(set(dock["mol_id"]))].reset_index(drop=True)
    dock = dock.set_index("mol_id").loc[lib["mol_id"]].reset_index()
    print(f"[data] 对接库 {len(lib)} 个分子（活性 {int(lib['label'].sum())}，"
          f"基准率 {lib['label'].mean():.1%}），有效对接 {len(dock)} 个")

    # ---- 第 1 步：2D 粗筛 ----
    splits = json.loads(Path(args.split).read_text())
    refs = splits["reference_actives"]  # 10k 库里的行号
    lib10k = pd.read_csv(vc.DATA_DIR / "drd3_library_10k.csv")
    ref_smiles = lib10k["smiles"].iloc[refs].tolist()
    X = vc.morgan_fp_matrix(lib["smiles"].tolist(), radius=2, n_bits=2048)
    R = vc.morgan_fp_matrix(ref_smiles, radius=2, n_bits=2048, verbose=False)
    sim = vc.tanimoto_cpu(X, R, chunk=2048).max(axis=1)
    lib["tanimoto_to_refs"] = sim
    lib["vina_affinity"] = dock["affinity"].to_numpy()
    y = lib["label"].to_numpy()

    n = len(lib)
    n_keep = max(1, int(round(n * args.top_frac)))
    order_2d = np.argsort(-lib["tanimoto_to_refs"].to_numpy(), kind="stable")
    order_dock = np.argsort(lib["vina_affinity"].to_numpy(), kind="stable")
    chosen = order_2d[:n_keep]

    # ---- 第 2 步：成本与富集对比 ----
    n_act = int(y.sum())
    n_zero = n_act == 0
    n_oneclass = n_zero or n_act == n
    if n_oneclass:
        print(
            "[warning] 当前 CSV 只有一类分子（活性 %d / 总数 %d）：富集与召回指标没有统计意义，"
            "仅用于验证流程能跑通。等预计算包含诱饵后再看数字。" % (n_act, n)
        )
    recall_2d = float(y[chosen].sum() / n_act) if n_act else float("nan")
    recall_random = float(n_keep / n)  # 随机抽同样数量的期望召回
    recall_dock_top = float(y[order_dock[:n_keep]].sum() / n_act) if n_act else float("nan")

    top_all = order_dock[: args.topk]
    top_pipe = chosen[np.argsort(-lib["vina_affinity"].to_numpy()[chosen], kind="stable")][: args.topk]
    hits_all = int(y[top_all].sum())
    hits_pipe = int(y[top_pipe].sum())
    overlap = len(set(top_all.tolist()) & set(top_pipe.tolist()))
    efs = {
        "ef1": vc.enrichment_factor(y, lib["vina_affinity"].to_numpy(), 0.01),
        "ef5": vc.enrichment_factor(y, lib["vina_affinity"].to_numpy(), 0.05),
        "roc_auc": vc.roc_auc(y, lib["vina_affinity"].to_numpy()),
        "ef1_2d": vc.enrichment_factor(y, lib["tanimoto_to_refs"].to_numpy(), 0.01),
        "ef5_2d": vc.enrichment_factor(y, lib["tanimoto_to_refs"].to_numpy(), 0.05),
        "roc_auc_2d": vc.roc_auc(y, lib["tanimoto_to_refs"].to_numpy()),
    }

    # 单次对接耗时（教师预计算时实测的墙钟，含进程启动与网格计算）
    per_dock_s = float(pd.to_numeric(dock.get("wall_s", pd.Series(dtype=float)), errors="coerce").median())
    if not np.isfinite(per_dock_s):
        per_dock_s = float("nan")
    cost_full = n * per_dock_s
    cost_pipe = n_keep * per_dock_s

    rows = [
        {
            "策略": "随机抽 10% 对接",
            "对接次数": n_keep,
            "占全库对接成本": f"{n_keep / n:.0%}",
            "活性召回": round(recall_random, 3),
            "top20 命中": round(args.topk * recall_random, 2),
            "说明": "下界参考，期望值",
        },
        {
            "策略": "2D 粗筛 top 10%（不对接）",
            "对接次数": 0,
            "占全库对接成本": "0%",
            "活性召回": round(recall_2d, 3),
            "top20 命中": "—",
            "说明": "纯 2D 特征的召回能力",
        },
        {
            "策略": f"2D 粗筛 10% → 只对接这 {n_keep} 个",
            "对接次数": n_keep,
            "占全库对接成本": f"{n_keep / n:.0%}",
            "活性召回": round(float(y[chosen].sum() / n_act), 3) if n_act else float("nan"),
            "top20 命中": hits_pipe,
            "说明": "课堂主推方案：粗筛后精排",
        },
        {
            "策略": f"全库对接（{n} 个）",
            "对接次数": n,
            "占全库对接成本": "100%",
            "活性召回": 1.0,
            "top20 命中": hits_all,
            "说明": "基线：成本上限，效果上限",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "strategy_comparison.csv", index=False)
    print("\n[策略对比]")
    print(table.to_string(index=False))

    print(f"\n[富集指标] 对接打分：ROC-AUC {efs['roc_auc']:.3f}，EF1% {efs['ef1']:.2f}，EF5% {efs['ef5']:.2f}")
    print(f"[富集指标] 2D 相似性：ROC-AUC {efs['roc_auc_2d']:.3f}，EF1% {efs['ef1_2d']:.2f}，EF5% {efs['ef5_2d']:.2f}")
    print(f"[top-{args.topk}] 全库对接命中 {hits_all} 个活性；粗筛+精排命中 {hits_pipe} 个；两者重合 {overlap}/{args.topk}")
    if np.isfinite(per_dock_s):
        print(
            f"[成本] 单次对接（vina，教师预计算实测中位数）{per_dock_s:.1f} s；"
            f"全库 {cost_full / 60:.0f} 分钟，粗筛方案 {cost_pipe / 60:.1f} 分钟"
        )

    # ---- 图：富集曲线 + 2D vs 对接散点 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for label, score in (("docking all", lib["vina_affinity"].to_numpy()), ("2D Tanimoto", lib["tanimoto_to_refs"].to_numpy())):
        xs, found = vc.enrichment_curve(y, score, n_points=200)
        ax.plot(xs * 100, found * 100, lw=2, label=label)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs * 100, xs * 100, "k--", lw=1, label="random")
    keep_pct = n_keep / n * 100
    ax.axvline(keep_pct, color="gray", ls=":", lw=1)
    ax.set_xlabel("fraction of library screened (%)")
    ax.set_ylabel("fraction of actives recovered (%)")
    ax.set_title(f"Enrichment (n={n}, coarse filter = top {args.top_frac:.0%})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    act = y == 1
    ax.scatter(lib.loc[~act, "tanimoto_to_refs"], lib.loc[~act, "vina_affinity"], s=6, alpha=0.4, label="decoys")
    ax.scatter(lib.loc[act, "tanimoto_to_refs"], lib.loc[act, "vina_affinity"], s=10, alpha=0.8, label="actives")
    ax.axvline(np.sort(lib["tanimoto_to_refs"].to_numpy())[::-1][n_keep - 1], color="gray", ls=":", lw=1)
    ax.set_xlabel("ECFP4 max Tanimoto to reference actives")
    ax.set_ylabel("vina affinity (kcal/mol)")
    ax.set_title("2D score vs docking score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_plot.png", dpi=150)
    plt.close(fig)

    summary = {
        "demo": "demo_pipeline",
        "n_library": n,
        "n_actives": n_act,
        "baseline_active_rate": round(float(y.mean()), 4),
        "top_frac": args.top_frac,
        "n_coarse_kept": n_keep,
        "recall_coarse_2d": round(recall_2d, 4),
        "recall_random_expected": round(recall_random, 4),
        "topk": args.topk,
        "topk_hits_full_docking": hits_all,
        "topk_hits_coarse_then_docking": hits_pipe,
        "topk_overlap": overlap,
        "metrics_docking": efs,
        "per_docking_seconds_median": round(per_dock_s, 2) if np.isfinite(per_dock_s) else None,
        "cost_full_minutes": round(cost_full / 60, 1) if np.isfinite(per_dock_s) else None,
        "cost_coarse_minutes": round(cost_pipe / 60, 1) if np.isfinite(per_dock_s) else None,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    vc.save_json(summary, out_dir / "pipeline_summary.json")
    print(f"\n[output] {out_dir}/strategy_comparison.csv")
    print(f"[output] {out_dir}/pipeline_plot.png")
    print(f"[output] {out_dir}/pipeline_summary.json")
    print(f"[total] 学生这一步 {vc.fmt_secs(summary['runtime_s'])}（不含教师预计算）")


if __name__ == "__main__":
    main()
