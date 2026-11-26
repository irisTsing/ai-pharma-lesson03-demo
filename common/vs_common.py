"""虚拟筛选 demo 的公共函数：数据读取、指纹、Tanimoto、富集指标、设备选择。

四个 demo 共用本模块。所有 demo 脚本都从 demos/demo_vs/common 导入。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
COMMON_DIR = Path(__file__).resolve().parent
DEMO_VS_DIR = COMMON_DIR.parent
DATA_DIR = DEMO_VS_DIR / "data"
TARGET = "drd3"


def set_matplotlib_cache() -> None:
    """matplotlib 在无写权限的 HOME 下会刷警告，统一指到 demos/.mplcache。"""
    os.environ.setdefault("MPLCONFIGDIR", str(DEMO_VS_DIR.parent / ".mplcache"))


# ----------------------------------------------------------------------------
# 设备选择：GPU 优先，失败回退 CPU，并把选择理由打印出来
# ----------------------------------------------------------------------------
def pick_device(prefer: str = "auto", require_gpu: bool = False) -> str:
    """返回 'cuda' 或 'cpu'。

    prefer: auto / cuda / cpu。auto 时优先 cuda，没有 GPU 就回退 cpu。
    require_gpu=True 且没有 GPU 时直接报错（用于必须在 gpu1 上跑的场合）。
    """
    if prefer not in ("auto", "cuda", "cpu"):
        raise SystemExit(f"--device 只能是 auto/cuda/cpu，收到 {prefer!r}")
    if prefer == "cpu":
        return "cpu"

    torch = _import_torch(optional=True)
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if has_cuda:
        return "cuda"
    if require_gpu:
        raise SystemExit(
            "要求 GPU 但当前机器上 torch.cuda.is_available() 为 False。\n"
            "  排查：1) nvidia-smi 是否可用；2) 是否在 GPU 节点（gpu1）上；\n"
            "        3) 环境是否装了 CUDA 版 torch（python -c 'import torch;print(torch.version.cuda)'）。\n"
            "  如果只想做 CPU 回退验证，去掉 --require-gpu 或加 --device cpu。"
        )
    print("[device] 未检测到可用 CUDA，自动回退 CPU（结果一致，只是慢一些）。")
    return "cpu"


def _import_torch(optional: bool = False):
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # pragma: no cover - 只在无 torch 环境触发
        if optional:
            return None
        raise SystemExit(f"导入 torch 失败：{exc}")


def env_report(device: str) -> dict:
    """打印并返回运行环境信息，写进 README / results。"""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "numpy": np.__version__,
    }
    try:
        import rdkit
        info["rdkit"] = rdkit.__version__
    except Exception:
        pass
    torch = _import_torch(optional=True)
    if torch is not None:
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["gpu_total_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------
# DUD-E 数据读取与化合物库构建
# ----------------------------------------------------------------------------
def read_ism(path: str | Path) -> list[dict]:
    """读 DUD-E 的 .ism 文件。

    格式：actives 每行 `SMILES DUD-E_ID CHEMBL_ID`，decoys 每行 `SMILES ZINC_ID`。
    返回 [{smiles, dud_id, mol_id, label}]。
    """
    path = Path(path)
    label = 1 if "active" in path.name else 0
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smiles = parts[0]
        dud_id = parts[1] if len(parts) > 1 else ""
        mol_id = parts[2] if len(parts) > 2 else dud_id
        rows.append({"smiles": smiles, "dud_id": dud_id, "mol_id": mol_id, "label": label})
    return rows


def load_library(path: str | Path) -> dict:
    """读 library CSV，返回列式 dict：smiles / label / mol_id。"""
    import pandas as pd

    df = pd.read_csv(path)
    return {
        "smiles": df["smiles"].tolist(),
        "label": df["label"].to_numpy(dtype=np.int64),
        "mol_id": df["mol_id"].astype(str).tolist(),
        "df": df,
    }


def build_library(
    target_dir: str | Path,
    n_decoys: int = 9520,
    seed: int = 42,
    out_csv: str | Path | None = None,
) -> "object":
    """从 DUD-E 原始文件构建筛选库：全部 actives + 固定随机种子抽样的 decoys。

    先统一用 RDKit 规范化 SMILES 并去重，保证每次生成同一份库。
    """
    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    target_dir = Path(target_dir)
    actives = read_ism(target_dir / "actives_final.ism")
    decoys = read_ism(target_dir / "decoys_final.ism")

    def canonical(rows):
        out, seen = [], set()
        for r in rows:
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            out.append({"smiles": can, **{k: r[k] for k in ("dud_id", "mol_id", "label")}})
        return out

    actives = canonical(actives)
    decoys = canonical(decoys)
    if n_decoys is not None and n_decoys < len(decoys):
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(decoys), size=n_decoys, replace=False))
        decoys = [decoys[i] for i in idx]

    rows = actives + decoys
    df = pd.DataFrame(rows)
    df.insert(0, "idx", np.arange(len(df)))
    df = df[["idx", "mol_id", "smiles", "label", "dud_id"]]
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


# ----------------------------------------------------------------------------
# 指纹
# ----------------------------------------------------------------------------
def morgan_fp_matrix(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_jobs: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """ECFP4（Morgan, radius=2）二进制指纹矩阵，uint8 0/1，形状 (n, n_bits)。"""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    n = len(smiles)
    X = np.zeros((n, n_bits), dtype=np.uint8)
    t0 = time.perf_counter()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        bits = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
        X[i] = bits
    if verbose:
        print(f"[fp] ECFP4 {n} 分子 × {n_bits} bit，耗时 {time.perf_counter() - t0:.2f} s")
    return X


# ----------------------------------------------------------------------------
# Tanimoto：CPU（numpy 分块矩阵乘）与 GPU（torch 分块矩阵乘）
# ----------------------------------------------------------------------------
def tanimoto_cpu(
    X: np.ndarray,
    Q: np.ndarray | None = None,
    chunk: int = 1024,
    dtype=np.float32,
) -> np.ndarray:
    """返回 X 与 Q 的 Tanimoto 矩阵，形状 (len(X), len(Q))；Q 缺省为 X 自身。

    二进制向量用 |a∩b| / (|a|+|b|-|a∩b|)，分子分母都由矩阵乘得到。
    """
    if Q is None:
        Q = X
    Xf = np.asarray(X, dtype=dtype)
    Qf = np.asarray(Q, dtype=dtype)
    q_norm = Qf.sum(axis=1)
    out = np.empty((Xf.shape[0], Qf.shape[0]), dtype=dtype)
    for s in range(0, Xf.shape[0], chunk):
        e = min(s + chunk, Xf.shape[0])
        block = Xf[s:e]
        inter = block @ Qf.T
        denom = block.sum(axis=1)[:, None] + q_norm[None, :] - inter
        out[s:e] = np.where(denom > 0, inter / np.maximum(denom, 1e-12), 0.0)
    return out


def tanimoto_gpu(
    X: np.ndarray,
    Q: np.ndarray | None = None,
    chunk: int = 2048,
    device: str = "cuda",
    report_memory: bool = True,
) -> np.ndarray:
    """GPU 版 Tanimoto，按行分块把指纹矩阵搬进显存，峰值显存 < 1 GB。

    注意：Ampere 显卡的 TF32 只有 10 位尾数，累加 2048 个 1 会溢出精度，
    这里显式关闭 TF32，保证与 CPU 结果逐位一致（误差 < 1e-6）。
    """
    torch = _import_torch()
    if Q is None:
        Q = X
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    Qt = torch.from_numpy(np.asarray(Q, dtype=np.float32)).to(device)
    q_norm = Qt.sum(dim=1)
    out = torch.empty((Xt.shape[0], Qt.shape[0]), dtype=torch.float32, device=device)
    for s in range(0, Xt.shape[0], chunk):
        e = min(s + chunk, Xt.shape[0])
        block = Xt[s:e]
        inter = block @ Qt.T
        denom = block.sum(dim=1, keepdim=True) + q_norm.unsqueeze(0) - inter
        out[s:e] = torch.where(denom > 0, inter / denom.clamp_min(1e-12), torch.zeros_like(denom))
    if report_memory and device.startswith("cuda"):
        torch.cuda.synchronize()
        print(
            f"[gpu] 峰值显存 {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB，"
            f"Tanimoto 计算耗时 {time.perf_counter() - t0:.3f} s"
        )
    return out.cpu().numpy()


def max_similarity_to_refs(fp: np.ndarray, ref_idx: list[int], device: str = "cpu") -> np.ndarray:
    """每个分子与参考分子的最大 Tanimoto 相似度。"""
    Q = fp[ref_idx]
    sim = tanimoto_gpu(fp, Q, device=device) if device == "cuda" else tanimoto_cpu(fp, Q)
    return sim.max(axis=1)


# ----------------------------------------------------------------------------
# 评价指标
# ----------------------------------------------------------------------------
def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """单类样本（全活性或全诱饵）时 ROC-AUC 无定义，返回 nan 并提示，不让流程崩掉。

    教师用“只跑了一部分”的预计算结果试跑时会出现这种情况，属于流程验证场景，
    此时富集指标没有统计意义。
    """
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        print("[metrics] 警告：样本只有一类，ROC-AUC 无定义，返回 nan（部分预计算数据仅用于流程验证）")
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score))


def enrichment_factor(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    """EF(frac) = 前 frac 比例里的活性命中率 / 全库活性比例。"""
    y = np.asarray(y)
    n = len(y)
    n_act = int(y.sum())
    if n == 0 or n_act == 0 or n_act == n:
        print(f"[metrics] 警告：基准率为 0 或 100%，EF{frac:.0%} 无统计意义，返回 nan")
        return float("nan")
    k = max(1, int(round(n * frac)))
    order = np.argsort(-score, kind="stable")
    hits = int(y[order[:k]].sum())
    base = n_act / n
    return float((hits / k) / base)


def top_k_hits(y: np.ndarray, score: np.ndarray, k: int) -> int:
    order = np.argsort(-score, kind="stable")
    return int(np.asarray(y)[order[:k]].sum())


def enrichment_curve(y: np.ndarray, score: np.ndarray, n_points: int = 200):
    """返回 (screened_fraction, found_actives_fraction)，用于画富集曲线。"""
    y = np.asarray(y)
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    n = len(y)
    n_act = max(1, int(ys.sum()))
    xs = np.linspace(0, 1, n_points)
    found = np.array([ys[: max(1, int(round(x * n)))].sum() for x in xs]) / n_act
    return xs, found


def summarize_scores(y: np.ndarray, score: np.ndarray, topk=(20, 50, 100)) -> dict:
    return {
        "n": int(len(y)),
        "n_actives": int(np.asarray(y).sum()),
        "roc_auc": roc_auc(y, score),
        "ef1": enrichment_factor(y, score, 0.01),
        "ef5": enrichment_factor(y, score, 0.05),
        "ef10": enrichment_factor(y, score, 0.10),
        "topk_hits": {f"top{k}": top_k_hits(y, score, k) for k in topk},
    }


# ----------------------------------------------------------------------------
# 划分：随机分层划分 与 Murcko scaffold 划分
# ----------------------------------------------------------------------------
def random_split(y: np.ndarray, frac_test: float = 0.2, seed: int = 2024):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for label in (0, 1):
        idx = np.where(np.asarray(y) == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * frac_test))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def scaffold_split(y: np.ndarray, smiles: list[str], frac_test: float = 0.2, seed: int = 2024):
    """按 Murcko scaffold 分组做标签分层的 8:2 划分，同一骨架只落在一侧。

    做法：先按骨架分组；再按类别分别设定测试集配额（活性 20%、诱饵 20%）；
    骨架组按大小从大到小尝试放入测试集，放进去会超出配额就放入训练集。
    这样既保证骨架不跨侧（避免训练集泄漏），又不会因为某类骨架特别大
    而把某一类全部挤到训练集（DUD-E DRD3 的诱饵里有 146 个分子共享同一骨架）。
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    y = np.asarray(y)
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            groups.setdefault(f"__bad__{i}", []).append(i)
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        groups.setdefault(scaf, []).append(i)

    quota = {lab: int(round(int((y == lab).sum()) * frac_test)) for lab in (0, 1)}
    got = {0: 0, 1: 0}
    keys = list(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)  # 先打乱，再按大小排序，保证同尺寸组的顺序也是随机的
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    test_idx: list[int] = []
    train_idx: list[int] = []
    for k in keys:
        idx = groups[k]
        n1 = int(y[idx].sum())
        n0 = len(idx) - n1
        fits = got[1] + n1 <= quota[1] and got[0] + n0 <= quota[0]
        if fits and (got[0] + got[1]) < int(round(len(y) * frac_test)):
            test_idx.extend(idx)
            got[1] += n1
            got[0] += n0
        else:
            train_idx.extend(idx)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def describe_split(y, train_idx, test_idx, name="") -> dict:
    y = np.asarray(y)
    return {
        "name": name,
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
        "train_actives": int(y[train_idx].sum()),
        "test_actives": int(y[test_idx].sum()),
        "train_active_rate": float(y[train_idx].mean()),
        "test_active_rate": float(y[test_idx].mean()),
    }


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_secs(sec: float) -> str:
    return f"{sec:.1f} s" if sec < 60 else f"{sec / 60:.1f} min"


def load_box(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
