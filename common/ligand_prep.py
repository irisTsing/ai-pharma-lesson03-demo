#!/usr/bin/env python
"""从 DUD-E 附带的 3D mol2 文件里取出指定分子的构象，并用 Open Babel 转成对接用的 PDBQT。

DUD-E 的补充数据里有 actives_final.mol2.gz / decoys_final.mol2.gz，
每个分子带 3D 坐标（部分分子有多个构象，取第一个）。SMILES 只有 2D 信息，
所以对接必须用这里的 3D 构象，或者用 obabel --gen3d 现场生成（慢且不稳定）。

用法（在 demos/demo_vs 下）：
    ../venv_chemprop/bin/python common/ligand_prep.py \
        --library data/drd3_dock_library.csv \
        --out-dir data/dock_ligands

产出：
    <out-dir>/mol2/<id>.mol2       从 DUD-E 抽出的 3D 构象
    <out-dir>/pdbqt/<id>.pdbqt     obabel -p 7.4 加氢后的配体
    <out-dir>/prep_summary.csv     每个分子的准备状态与分子式核对结果
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vs_common as vc

OBABEL = "/usr/local/bin/obabel"
CHEM_TYPES = {"C.3", "C.2", "C.1", "C.ar", "C.cat", "N.3", "N.2", "N.1", "N.ar", "N.am",
              "N.pl3", "N.4", "O.3", "O.2", "O.co2", "S.3", "S.2", "S.O", "S.O2", "P.3",
              "F", "Cl", "Br", "I", "H"}


def stream_mol2_blocks(path: Path, wanted: set[str], key_fn):
    """流式扫描多分子 mol2.gz，只保留 wanted 里出现过的分子（每个只取第一个构象）。

    key_fn 把 mol2 标题转成库里的编号（诱饵的 ZINC 号要转成 C 号）。
    """
    found: dict[str, str] = {}
    cur_name = None
    cur_lines: list[str] = []
    blocks: list[tuple[str, str]] = []

    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>MOLECULE"):
                if cur_name is not None:
                    key = key_fn(cur_name)
                    if key in wanted and key not in found:
                        found[key] = "".join(cur_lines)
                cur_name = None
                cur_lines = [line]
                continue
            if cur_name is None and len(cur_lines) == 1:
                cur_name = line.strip()
                cur_lines.append(line)
                continue
            cur_lines.append(line)
    if cur_name is not None:
        key = key_fn(cur_name)
        if key in wanted and key not in found:
            found[key] = "".join(cur_lines)
    blocks = list(found.items())
    return blocks


def normalize_key(title: str, label: int) -> str:
    """DUD-E mol2 标题与 .ism 里的编号对应关系。

    活性分子：mol2 标题就是 CHEMBL 号，与 .ism 第三列一致；
    诱饵：.ism 第二列是 C02651967 这种编号，mol2 标题是 ZINC02651967，
    对应关系是 C + 标题去掉 "ZINC" 前缀后的数字部分。
    """
    title = title.strip()
    if label == 1:
        return title
    if title.upper().startswith("ZINC"):
        return "C" + title[4:]
    return title


_TAUTOMER_ENUM = None


def connectivity_key(mol) -> str:
    """中性化 + 互变异构归一 + 去立体后的 canonical SMILES。

    DUD-E 的 3D 构象按 pH 7.4 质子化（例如叔胺变成 [NH+]），
    而 .ism 里的 SMILES 是中性形式，直接比较必然不同，所以先中和电荷；
    少数分子（如吡唑）的 N-H 位置不同，属于互变异构，再用 RDKit 的
    标准化互变异构体统一，避免把同一分子误判成不同分子。
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    global _TAUTOMER_ENUM
    if mol is None:
        return ""
    try:
        mol = Chem.RemoveHs(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        if _TAUTOMER_ENUM is None:
            _TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
        mol = _TAUTOMER_ENUM.Canonicalize(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def connectivity_of_smiles(smiles: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return connectivity_key(Chem.MolFromSmiles(smiles))


def connectivity_of_mol2_block(block: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromMol2Block(block, sanitize=True, removeHs=False)
    except Exception:
        mol = None
    return connectivity_key(mol)


def convert_one(mol2_path: Path, pdbqt_path: Path) -> tuple[bool, str]:
    """obabel 2.4 的选项要写在输入文件之后；-p 7.4 按 pH 7.4 加氢。"""
    cmd = [OBABEL, str(mol2_path), "-O", str(pdbqt_path), "-p", "7.4"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = pdbqt_path.exists() and "1 molecule converted" in proc.stderr
    return ok, proc.stderr.strip()[-300:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(vc.DATA_DIR / "drd3_dock_library.csv"))
    ap.add_argument("--target-dir", default=str(vc.DATA_DIR / "drd3"))
    ap.add_argument("--out-dir", default=str(vc.DATA_DIR / "dock_ligands"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只准备前 N 个分子（调试用）")
    args = ap.parse_args()

    lib = vc.load_library(args.library)
    df = lib["df"]
    if args.limit:
        df = df.head(args.limit).copy()
    target_dir = Path(args.target_dir)
    out_dir = Path(args.out_dir)
    mol2_dir = out_dir / "mol2"
    pdbqt_dir = out_dir / "pdbqt"
    mol2_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    actives = df[df["label"] == 1]
    decoys = df[df["label"] == 0]
    print(f"[prep] 需要 3D 构象：actives={len(actives)} decoys={len(decoys)}")

    records: dict[str, dict] = {}

    def collect(sub: pd.DataFrame, gz_name: str, label: int) -> None:
        wanted = set(sub["mol_id"].astype(str))
        path = target_dir / gz_name
        if not path.exists():
            raise SystemExit(f"缺少 {path}，先运行 data/download_data.sh")
        blocks = stream_mol2_blocks(path, wanted, lambda t, lab=label: normalize_key(t, lab))
        print(f"[prep] {gz_name}: 命中 {len(blocks)}/{len(wanted)}")
        for title, block in blocks:
            key = normalize_key(title, label)
            if key not in wanted:
                continue
            m2 = mol2_dir / f"{key}.mol2"
            m2.write_text(block, encoding="utf-8")
            records.setdefault(key, {})["mol2_path"] = str(m2)

    collect(actives, "actives_final.mol2.gz", 1)
    collect(decoys, "decoys_final.mol2.gz", 0)

    # 分子式核对：库里的 SMILES 与 mol2 的 3D 结构必须是同一分子（允许立体化学差异）
    rows = []
    for _, r in df.iterrows():
        key = str(r["mol_id"])
        rec = {"mol_id": key, "label": int(r["label"]), "smiles": r["smiles"]}
        rec["has_mol2"] = key in records
        if key not in records:
            rec["status"] = "missing_mol2"
            rows.append(rec)
            continue
        rec["mol2_path"] = records[key]["mol2_path"]
        c_smi = connectivity_of_smiles(r["smiles"])
        block = Path(records[key]["mol2_path"]).read_text(encoding="utf-8")
        c_mol2 = connectivity_of_mol2_block(block)
        rec["conn_smiles"] = c_smi
        rec["conn_mol2"] = c_mol2
        rec["connectivity_match"] = bool(c_smi and c_smi == c_mol2)
        rec["status"] = "ok" if rec["connectivity_match"] else "connectivity_mismatch"
        rows.append(rec)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    print(f"[prep] 分子式核对通过 {len(ok_rows)}/{len(rows)}")
    if len(ok_rows) < len(rows):
        bad = [r for r in rows if r["status"] != "ok"]
        print(f"[prep] 未通过示例：{[(b['mol_id'], b['status']) for b in bad[:5]]}")
        for b in bad[:3]:
            print(f"   {b['mol_id']}: 库={b.get('conn_smiles')} / 3D={b.get('conn_mol2')}")

    # 并行转 pdbqt
    jobs = [(Path(r["mol2_path"]), pdbqt_dir / f"{r['mol_id']}.pdbqt") for r in ok_rows]

    def run(job):
        mol2_path, pdbqt_path = job
        try:
            ok, msg = convert_one(mol2_path, pdbqt_path)
        except Exception as exc:  # pragma: no cover
            return mol2_path.stem, False, str(exc)
        return mol2_path.stem, ok, msg

    status: dict[str, tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (key, ok, msg) in enumerate(pool.map(run, jobs), 1):
            status[key] = (ok, msg)
            if i % 200 == 0:
                print(f"[prep] pdbqt {i}/{len(jobs)}")

    for r in rows:
        key = r["mol_id"]
        ok, msg = status.get(key, (False, "not_converted"))
        if r["status"] == "ok":
            r["status"] = "ok" if ok else "obabel_failed"
            r["pdbqt"] = str(pdbqt_dir / f"{key}.pdbqt") if ok else ""
            if not ok:
                r["obabel_msg"] = msg
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "prep_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary = summary.drop(columns=[c for c in ("conn_smiles", "conn_mol2") if c in summary.columns])
    n_ok = int((summary["status"] == "ok").sum())
    print(f"[prep] 完成：可用 {n_ok}/{len(summary)}，汇总 -> {summary_path}")
    if n_ok < len(summary):
        print(summary["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
