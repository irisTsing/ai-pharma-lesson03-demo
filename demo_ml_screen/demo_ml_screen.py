#!/usr/bin/env python
"""demo 2：用 ECFP4 指纹训练机器学习打分器，与 demo 1 的相似性检索对照。

两条模型线：
1. CPU：scikit-learn 随机森林（300 棵树，class_weight=balanced_subsample）；
2. GPU：torch 多层感知机（2048 -> 512 -> 128 -> 1，ReLU + Dropout + Adam），
   同一份指纹，记录单轮耗时与峰值显存；没有 GPU 时在 CPU 上跑同一个模型。

划分：主口径是 Murcko scaffold 划分（同一骨架不跨训练/测试），
另外跑一遍随机划分作对照，用测试集 ROC-AUC 的差距说明“随机划分会高估性能”。

评价：测试集上 EF1%、EF5%、ROC-AUC、top-k 命中数，与 demo 1 的 Tanimoto 打分同表对比。

运行：
    ../venv_chemprop/bin/python demo_ml_screen/demo_ml_screen.py --device auto
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
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-jobs", type=int, default=4, help="随机森林线程数；课堂 50 人共用节点时不要设成 -1")
    ap.add_argument("--torch-threads", type=int, default=4, help="torch CPU 线程数（GPU 时不起作用）")
    ap.add_argument("--out-dir", default=str(DEMO_DIR / "results"))
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 随机森林（CPU）
# ---------------------------------------------------------------------------
def make_rf(n_trees: int, n_jobs: int = 4):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=2024,
    )


def train_rf(X_train, y_train, X_all, n_trees: int, n_jobs: int = 4):
    """返回 (全库概率, 训练耗时, 全库推理耗时)。"""
    clf = make_rf(n_trees, n_jobs)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_all = clf.predict_proba(X_all)[:, 1]
    infer_s = time.perf_counter() - t0
    return p_all, train_s, infer_s


# ---------------------------------------------------------------------------
# 多层感知机（torch，GPU 或 CPU）
# ---------------------------------------------------------------------------
def build_mlp(in_dim: int, hidden: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )


def train_mlp(X_train, y_train, X_all, device: str, args):
    import torch
    import torch.nn as nn

    torch.manual_seed(2024)
    np.random.seed(2024)
    if device == "cuda":
        torch.cuda.manual_seed_all(2024)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_mlp(X_train.shape[1], args.hidden).to(device)
    xtr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1).to(device)
    n_pos = ytr.sum().clamp_min(1)
    pos_weight = torch.tensor([(ytr.numel() - ytr.sum()) / n_pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = xtr.shape[0]
    epoch_times = []
    first_loss = None
    model.train()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s : s + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        if epoch == 0:
            first_loss = total / n
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [mlp] epoch {epoch + 1:3d}/{args.epochs}  loss {total / n:.4f}  {epoch_times[-1] * 1000:.0f} ms")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = []
        for s in range(0, len(X_all), 4096):
            xb = torch.from_numpy(X_all[s : s + 4096].astype(np.float32)).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy().ravel())
        if device == "cuda":
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - t0
    p_all = np.concatenate(out)
    timing = {
        "epochs": args.epochs,
        "first_epoch_s": round(epoch_times[0], 3),
        "mean_epoch_s": round(float(np.mean(epoch_times)), 3),
        "total_train_s": round(float(np.sum(epoch_times)), 2),
        "peak_gpu_mem_gb": round(peak_gb, 3) if peak_gb else None,
        "infer_library_s": round(infer_s, 3),
        "first_loss": round(first_loss, 4) if first_loss is not None else None,
    }
    return p_all, timing


def main() -> None:
    args = parse_args()
    vc.set_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_curve

    device = vc.pick_device(args.device, require_gpu=args.require_gpu)
    if device == "cpu":
        import torch

        torch.set_num_threads(args.torch_threads)
        print(f"[env] torch CPU 线程数 {torch.get_num_threads()}；随机森林线程数 {args.n_jobs}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    print("=" * 70)
    print("demo 2：ECFP4 + 机器学习打分器（随机森林 / 多层感知机）")
    print("=" * 70)
    env = vc.env_report(device)
    print(f"[env] device={device}  torch {env.get('torch')}  sklearn {env.get('sklearn')}")
    if device == "cuda":
        print(f"[env] GPU {env.get('gpu_name')}  显存 {env.get('gpu_total_mem_gb')} GB")

    lib = vc.load_library(args.library)
    splits = json.loads(Path(args.split).read_text())
    y = lib["label"]
    smiles = lib["smiles"]
    X = vc.morgan_fp_matrix(smiles, radius=2, n_bits=2048).astype(np.float32)
    refs = splits["reference_actives"]

    # demo 1 的相似性打分（同一份参考分子，同一批评价分子）
    sim_score = vc.tanimoto_cpu(X, X[refs], chunk=2048).max(axis=1)

    te_scaffold = np.array(splits["scaffold"]["test"])
    tr_scaffold = np.array(splits["scaffold"]["train"])

    def eval_on(name, score, test_idx, timing=None):
        y_test = y[test_idx]
        m = vc.summarize_scores(y_test, score[test_idx], topk=(20, 50, 100))
        m.update(
            {
                "method": name,
                "test_n": int(len(test_idx)),
                "test_actives": int(y_test.sum()),
                "baseline_active_rate": float(y_test.mean()),
            }
        )
        if timing:
            m["timing"] = timing
        return m

    results: dict[str, dict] = {}
    lib_scores: dict[str, np.ndarray] = {}
    for split_name in ("scaffold", "random"):
        tr = np.array(splits[split_name]["train"])
        te = np.array(splits[split_name]["test"])
        print(f"\n[split:{split_name}] train {len(tr)}（活性 {int(y[tr].sum())}）  test {len(te)}（活性 {int(y[te].sum())}）")

        p_rf, rf_train_s, rf_infer_s = train_rf(X[tr], y[tr], X, args.n_trees, args.n_jobs)
        rf_time = {"train_s": round(rf_train_s, 2), "infer_library_s": round(rf_infer_s, 3)}
        results[f"rf_{split_name}"] = eval_on("RandomForest", p_rf, te, rf_time)
        print(
            f"  [RF] 训练 {rf_train_s:.1f} s  测试集 ROC-AUC {results[f'rf_{split_name}']['roc_auc']:.3f}"
            f"  EF1% {results[f'rf_{split_name}']['ef1']:.2f}  EF5% {results[f'rf_{split_name}']['ef5']:.2f}"
        )

        p_mlp, mlp_time = train_mlp(X[tr], y[tr], X, device, args)
        results[f"mlp_{split_name}"] = eval_on("MLP", p_mlp, te, mlp_time)
        r = results[f"mlp_{split_name}"]
        print(
            f"  [MLP:{device}] 训练 {mlp_time['total_train_s']:.1f} s（{mlp_time['mean_epoch_s'] * 1000:.0f} ms/epoch）"
            f"  测试集 ROC-AUC {r['roc_auc']:.3f}  EF1% {r['ef1']:.2f}  EF5% {r['ef5']:.2f}"
            + (f"  峰值显存 {mlp_time['peak_gpu_mem_gb']:.3f} GB" if mlp_time["peak_gpu_mem_gb"] else "")
        )
        if split_name == "scaffold":
            lib_scores["rf"] = p_rf
            lib_scores["mlp"] = p_mlp

    results["similarity_ecfp4"] = eval_on("Tanimoto (demo 1)", sim_score, te_scaffold)
    results["random_baseline"] = {
        "method": "random",
        "roc_auc": 0.5,
        "ef1": 1.0,
        "ef5": 1.0,
        "ef10": 1.0,
        "topk_hits": {
            "top20": round(20 * float(y[te_scaffold].mean()), 2),
            "top50": round(50 * float(y[te_scaffold].mean()), 2),
            "top100": round(100 * float(y[te_scaffold].mean()), 2),
        },
        "test_n": int(len(te_scaffold)),
        "test_actives": int(y[te_scaffold].sum()),
    }

    # 对照表：所有方法在同一测试集（scaffold 划分的 2000 个分子）上评价
    rows = []
    for key, label in (
        ("random_baseline", "随机排序"),
        ("similarity_ecfp4", "Tanimoto 相似性（demo 1）"),
        ("rf_scaffold", "随机森林（scaffold 划分）"),
        ("mlp_scaffold", f"MLP（{device}，scaffold 划分）"),
        ("rf_random", "随机森林（随机划分）"),
        ("mlp_random", f"MLP（{device}，随机划分）"),
    ):
        m = results[key]
        rows.append(
            {
                "method": label,
                "roc_auc": round(m["roc_auc"], 4),
                "ef1": round(m["ef1"], 2),
                "ef5": round(m["ef5"], 2),
                "ef10": round(m["ef10"], 2),
                "top20_hits": m["topk_hits"]["top20"],
                "top50_hits": m["topk_hits"]["top50"],
                "top100_hits": m["topk_hits"]["top100"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison_table.csv", index=False)
    print("\n[对照表：scaffold 测试集 2000 个分子，其中 96 个活性]")
    print(table.to_string(index=False))

    # 图：富集曲线 + ROC
    plot_scores = {
        "Tanimoto (demo 1)": sim_score[te_scaffold],
        "RandomForest": lib_scores["rf"][te_scaffold],
        "MLP": lib_scores["mlp"][te_scaffold],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, sc in plot_scores.items():
        xs, found = vc.enrichment_curve(y[te_scaffold], sc, n_points=200)
        ax.plot(xs * 100, found * 100, lw=2, label=name)
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="random")
    ax.set_xlabel("fraction of library screened (%)")
    ax.set_ylabel("fraction of actives recovered (%)")
    ax.set_title("Enrichment curve (scaffold test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    aucs = {
        "Tanimoto (demo 1)": results["similarity_ecfp4"]["roc_auc"],
        "RandomForest": results["rf_scaffold"]["roc_auc"],
        "MLP": results["mlp_scaffold"]["roc_auc"],
    }
    for name, sc in plot_scores.items():
        fpr, tpr, _ = roc_curve(y[te_scaffold], sc)
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={aucs[name]:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve (scaffold test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "enrichment_ml.png", dpi=150)
    plt.close(fig)

    # 全库打分，供 demo 4 / 课后作业复用。
    # 注意：模型在 8000 个训练分子上拟合过，训练部分的打分偏乐观，不能直接当筛选结果汇报。
    pd.DataFrame(
        {
            "mol_id": lib["mol_id"],
            "smiles": smiles,
            "label": y,
            "rf_score": np.round(lib_scores["rf"], 6),
            "mlp_score": np.round(lib_scores["mlp"], 6),
            "tanimoto_max_to_refs": np.round(sim_score, 6),
        }
    ).to_csv(out_dir / "library_scores.csv", index=False)

    elapsed = time.perf_counter() - t_all
    result = {
        "demo": "demo_ml_screen",
        "device": device,
        "env": env,
        "params": vars(args),
        "splits": {k: splits[k]["summary"] for k in ("scaffold", "random")},
        "results": results,
        "comparison_table": rows,
        "timing_s": {"total": round(elapsed, 2)},
    }
    vc.save_json(result, out_dir / "metrics.json")
    print(f"\n[output] {out_dir}/comparison_table.csv")
    print(f"[output] {out_dir}/enrichment_ml.png")
    print(f"[output] {out_dir}/library_scores.csv")
    print(f"[total] 全流程 {vc.fmt_secs(elapsed)}")


if __name__ == "__main__":
    main()
