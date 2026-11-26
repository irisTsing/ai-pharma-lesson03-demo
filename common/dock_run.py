#!/usr/bin/env python
"""对接执行与结果解析：vina（CPU 参考实现）与 Uni-Dock（GPU 批量对接）。

设计要点：
- 单配体一次进程调用，用多进程并行；不用 vina 的 --batch（1.2.3 的 batch 选项
  要求特殊格式的多模型 PDBQT，教学场景不值得折腾，见 README 常见坑）。
- 打分统一从输出 pose 的 `REMARK VINA RESULT:` 行读取，四舍五入到 0.001 kcal/mol，
  这样 vina 与 Uni-Dock 的打分表可以直接对比。
- 每个配体记录墙钟耗时，供 README 写实测数字。

被 demo3（12 个配体，现场跑）和 demo4（教师预计算，后台跑）共用。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

VINA = "/usr/bin/vina"
OBABEL = "/usr/local/bin/obabel"
RESULT_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def box_args(box: dict) -> list[str]:
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    return [
        "--center_x", f"{cx:.3f}",
        "--center_y", f"{cy:.3f}",
        "--center_z", f"{cz:.3f}",
        "--size_x", f"{sx:.1f}",
        "--size_y", f"{sy:.1f}",
        "--size_z", f"{sz:.1f}",
    ]


def parse_pose_affinity(pdbqt_path: Path) -> float | None:
    """从对接 pose 里取第一个 MODEL 的 affinity（最优构象）。"""
    try:
        text = pdbqt_path.read_text(errors="replace")
    except OSError:
        return None
    m = RESULT_RE.search(text)
    return float(m.group(1)) if m else None


def _run_vina_one(job: dict) -> dict:
    """单个配体的 vina 调用。job 里带全部参数，便于多进程传递。"""
    lig = Path(job["ligand"])
    out = Path(job["out"])
    log = Path(job["log"])
    cmd = [
        job["vina"],
        "--receptor", job["receptor"],
        "--ligand", str(lig),
        *box_args(job["box"]),
        "--cpu", str(job["cpu"]),
        "--exhaustiveness", str(job["exhaustiveness"]),
        "--num_modes", str(job["num_modes"]),
        "--seed", str(job["seed"]),
        "--out", str(out),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=job.get("timeout", 600))
        wall = time.perf_counter() - t0
        log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        affinity = parse_pose_affinity(out) if out.exists() else None
        ok = proc.returncode == 0 and affinity is not None
        return {
            "ligand": lig.stem,
            "status": "ok" if ok else "failed",
            "affinity": affinity,
            "wall_s": round(wall, 2),
            "returncode": proc.returncode,
            "message": "" if ok else (proc.stderr.strip()[-300:] or "no affinity in pose"),
        }
    except subprocess.TimeoutExpired:
        return {
            "ligand": lig.stem,
            "status": "timeout",
            "affinity": None,
            "wall_s": round(time.perf_counter() - t0, 2),
            "returncode": -1,
            "message": "timeout",
        }


def run_vina_parallel(
    ligands: list[str | Path],
    receptor: str | Path,
    box: dict,
    out_dir: str | Path,
    cpu: int = 2,
    exhaustiveness: int = 8,
    num_modes: int = 1,
    workers: int = 4,
    seed: int = 2024,
    keep_poses: bool = True,
    tag: str = "vina",
    verbose: bool = True,
) -> pd.DataFrame:
    """多进程并行跑 vina。保持每个配体一个独立进程，按配体记录耗时。"""
    out_dir = Path(out_dir)
    pose_dir = out_dir / "poses"
    log_dir = out_dir / "logs"
    pose_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, lig in enumerate(ligands):
        lig = Path(lig)
        jobs.append(
            {
                "vina": VINA,
                "receptor": str(receptor),
                "ligand": str(lig),
                "out": str(pose_dir / f"{lig.stem}.pdbqt"),
                "log": str(log_dir / f"{lig.stem}.log"),
                "box": box,
                "cpu": cpu,
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
                # 每个配体用可复现的种子
                "seed": int(seed + i),
            }
        )

    rows = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_vina_one, j) for j in jobs]
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if verbose and (done % 25 == 0 or done == len(jobs)):
                el = time.perf_counter() - t_start
                print(f"[{tag}] {done}/{len(jobs)}  已用 {el:.0f} s  预计剩余 {el / done * (len(jobs) - done):.0f} s")
    total = time.perf_counter() - t_start
    df = pd.DataFrame(rows)
    df["method"] = tag
    df["wall_total_s"] = round(total, 2)
    df = df.sort_values("affinity", na_position="last").reset_index(drop=True)
    if not keep_poses:
        shutil.rmtree(pose_dir, ignore_errors=True)
    return df


# ---------------------------------------------------------------------------
# Uni-Dock（GPU）
# ---------------------------------------------------------------------------
def find_unidock() -> str | None:
    """找 unidock 可执行文件：先 PATH，再环境变量 UNIDOCK，最后几个常见路径。"""
    import os

    for cand in (os.environ.get("UNIDOCK"), shutil.which("unidock")):
        if cand and Path(cand).exists():
            return cand
    for cand in (
        "/public/software/unidock/bin/unidock",
        str(Path.home() / "miniconda3/envs/unidock_env/bin/unidock"),
    ):
        if Path(cand).exists():
            return cand
    return None


def gpu_selfcheck() -> tuple[bool, str]:
    """Uni-Dock 需要 NVIDIA GPU 与 compute capability >= 7.0（RTX 3090 是 8.6）。

    torch 能给出最完整的设备信息（名字、capability、显存），所以优先用 torch；
    torch 不可用时退回 nvidia-smi -L，只确认卡在不在。
    """
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] < 7:
                return False, f"显卡 compute capability {cap} < 7.0，Uni-Dock 不支持"
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"{name}（capability {cap[0]}.{cap[1]}，显存 {total:.0f} GB）"
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
            if proc.returncode == 0 and proc.stdout.strip():
                return True, proc.stdout.strip().splitlines()[0]
        except Exception:
            pass
    return False, "未检测到可用 NVIDIA GPU（nvidia-smi 不可用或 torch.cuda 不可用）"


def _parse_unidock_outputs(out_dir: Path, stems: list[str]) -> dict[str, float | None]:
    """从 Uni-Dock 的输出目录回收每个配体的最优打分。

    已核实（Uni-Dock v1.2.0，conda-forge 包）：
    - 输出文件命名为 `<配体文件名主干>_out.pdbqt`（输入 pdbqt 时）或 `_out.sdf`；
    - SDF 里的打分写在 `> <Uni-Dock RESULT>` 属性下，下一行形如 `ENERGY=-9.4  LOWER_BOUND=...`；
    - pdbqt 里与 vina 一样是 `REMARK VINA RESULT:`。
    为了兼容不同小版本的目录布局，这里递归搜索，不写死单层结构。
    """
    found: dict[str, Path] = {}
    for p in out_dir.rglob("*_out.*"):
        stem = p.name[: -len("_out" + p.suffix)]
        found.setdefault(stem, p)
    result: dict[str, float | None] = {}
    for stem in stems:
        path = found.get(stem)
        if path is None:
            result[stem] = None
            continue
        if path.suffix == ".pdbqt":
            result[stem] = parse_pose_affinity(path)
        else:
            result[stem] = parse_sdf_affinity(path)
    return result


def parse_sdf_affinity(sdf_path: Path) -> float | None:
    """解析 Uni-Dock 的 SDF 输出：> <Uni-Dock RESULT> 下一行的 ENERGY=。"""
    try:
        lines = sdf_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for i, line in enumerate(lines):
        if line.startswith("> <Uni-Dock RESULT>"):
            for j in range(i + 1, min(i + 4, len(lines))):
                m = re.search(r"ENERGY=(-?\d+(?:\.\d+)?)", lines[j])
                if m:
                    return float(m.group(1))
    # 少数构建版本仍写 REMARK VINA RESULT
    return parse_pose_affinity(sdf_path)


def run_unidock(
    ligands: list[str | Path],
    receptor: str | Path,
    box: dict,
    out_dir: str | Path,
    search_mode: str = "fast",
    scoring: str = "vina",
    num_modes: int = 1,
    seed: int = 2024,
    max_gpu_memory_mb: int = 0,
    extra_args: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """一次调用把多个配体一起交给 Uni-Dock，GPU 批量对接。

    max_gpu_memory_mb：Uni-Dock 默认占满显存，多人共用一张卡时必须设上限。
    8 卡 50 人场景下建议 2500~3000 MB（见 README 的并发方案）。
    """
    unidock = find_unidock()
    if unidock is None:
        raise RuntimeError(
            "找不到 unidock。先在 gpu1 上执行 demos/demo_vs/setup_gpu_env.sh 并把它的 bin 加进 PATH，"
            "或用 UNIDOCK=/path/to/unidock 指定；也可以加 --device cpu 走 vina 回退。"
        )
    ok, msg = gpu_selfcheck()
    if not ok:
        raise RuntimeError(f"GPU 自检失败：{msg}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lig_args = [str(p) for p in ligands]
    cmd = [
        unidock,
        "--receptor", str(receptor),
        "--gpu_batch", *lig_args,
        "--search_mode", search_mode,
        "--scoring", scoring,
        *box_args(box),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        "--dir", str(out_dir),
    ]
    if max_gpu_memory_mb:
        cmd += ["--max_gpu_memory", str(max_gpu_memory_mb)]
    if extra_args:
        cmd += list(extra_args)
    if verbose:
        print(f"[unidock] {unidock}  batch={len(lig_args)}  search_mode={search_mode}  scoring={scoring}")
        print(f"[unidock] 显存上限 {max_gpu_memory_mb or '不限制'} MB，输出目录 {out_dir}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    log_path = out_dir / "unidock.log"
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        combined = (proc.stdout + proc.stderr).lower()
        if "cudaerrornodevice" in combined or "cuda runtime error" in combined:
            raise RuntimeError(
                "Uni-Dock 报 CUDA 错误：当前进程看不到可用 GPU。\n"
                "  检查：nvidia-smi 是否正常；CUDA_VISIBLE_DEVICES 是否指向一张存在的卡；\n"
                "  或加 --device cpu 用 vina 回退。日志：" + str(log_path)
            )
        raise RuntimeError(f"Uni-Dock 返回码 {proc.returncode}，日志：{log_path}\n{proc.stderr.strip()[-600:]}")

    stems = [Path(p).stem for p in ligands]
    affinities = _parse_unidock_outputs(out_dir, stems)
    rows = []
    for stem in stems:
        aff = affinities.get(stem)
        rows.append(
            {
                "ligand": stem,
                "status": "ok" if aff is not None else "no_result",
                "affinity": aff,
                "wall_s": None,  # 批量执行，单配体耗时按平均值记
                "returncode": 0,
                "message": "" if aff is not None else "输出目录里没找到 _out.pdbqt/_out.sdf",
            }
        )
    df = pd.DataFrame(rows)
    df["method"] = f"unidock_{search_mode}"
    df["wall_total_s"] = round(wall, 2)
    df["wall_per_lig_s"] = round(wall / max(1, len(ligands)), 3)
    df["gpu_mem_limit_mb"] = max_gpu_memory_mb
    df = df.sort_values("affinity", na_position="last").reset_index(drop=True)
    return df


def summarize_compare(df_vina: pd.DataFrame, df_gpu: pd.DataFrame) -> pd.DataFrame:
    """把 vina（CPU）与 Uni-Dock（GPU）的结果按配体对齐，给出分数差与加速比。"""
    cols = ["ligand", "status", "affinity"]
    a = df_vina[cols].rename(columns={"status": "vina_status", "affinity": "vina_affinity"})
    b = df_gpu[cols].rename(columns={"status": "unidock_status", "affinity": "unidock_affinity"})
    m = a.merge(b, on="ligand", how="outer")
    m["delta_kcal"] = (m["unidock_affinity"] - m["vina_affinity"]).round(3)
    if "wall_per_lig_s" in df_gpu.columns:
        m["unidock_s_per_lig"] = df_gpu["wall_per_lig_s"].iloc[0]
    m["vina_s_per_lig"] = df_vina["wall_s"].mean()
    return m


def write_docking_table(df: pd.DataFrame, path: str | Path, box: dict, extra: dict | None = None) -> None:
    """保存打分表 CSV，并把盒子与运行参数写进同目录的 json。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = {"box": box, **(extra or {})}
    (path.parent / (path.stem + "_meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    # 自检：对 2 个配体跑一次小规模 vina，确认工具链可用
    here = Path(__file__).resolve().parent.parent
    box = json.loads((here / "data/drd3_box.json").read_text())
    ligs = sorted((here / "data/dock_ligands/pdbqt").glob("*.pdbqt"))[:2]
    df = run_vina_parallel(ligs, here / "data/drd3/receptor.pdbqt", box, here / ".scratch/selfcheck",
                           cpu=2, exhaustiveness=4, workers=2)
    print(df.to_string(index=False))
