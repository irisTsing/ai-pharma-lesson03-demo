#!/usr/bin/env python
"""从 vina 的日志目录增量汇总对接打分，生成/更新 full_docking.csv。

用途：教师预计算要跑几十分钟，学生不能等它全部跑完才能上课。
本脚本随时可以把“已经跑完的那些配体”汇总成 CSV，demo 4 就能先跑；
全部跑完后重新执行一次，得到完整 CSV（与 precompute_docking.py 的收尾逻辑等价）。

用法（在 demos/demo_vs 下）：
    ../venv_chemprop/bin/python demo_pipeline/collect_dock_logs.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
DEMO_VS_DIR = DEMO_DIR.parent
sys.path.insert(0, str(DEMO_VS_DIR / "common"))
import vs_common as vc  # noqa: E402

AFF_RE = re.compile(r"^\s+1\s+(-?\d+\.\d+)\s", re.MULTILINE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=str(DEMO_DIR / "results/dock_work/logs"))
    ap.add_argument("--library", default=str(vc.DATA_DIR / "drd3_dock_library.csv"))
    ap.add_argument("--out", default=str(DEMO_DIR / "results/full_docking.csv"))
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    lib = pd.read_csv(args.library)
    lib["mol_id"] = lib["mol_id"].astype(str)
    label = dict(zip(lib["mol_id"], lib["label"]))
    rows = []
    for log in sorted(log_dir.glob("*.log")):
        text = log.read_text(errors="replace")
        m = AFF_RE.search(text)
        if not m:
            continue
        aff = float(m.group(1))
        rows.append(
            {
                "mol_id": log.stem,
                "label": label.get(log.stem),
                "affinity": aff,
                "valid": aff < 0,
                "status": "ok" if aff < 0 else "invalid_positive_score",
                "method": "vina",
            }
        )
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[collect] {len(df)} 个配体有结果 -> {out}")
    print(f"[collect] 有效 {int(df['valid'].sum())}，正分（异常）{int((~df['valid']).sum())}")
    print(f"[collect] 覆盖活性分子 {int(df.loc[df['valid'], 'label'].sum())}/{int(lib['label'].sum())}")
    done = set(df["mol_id"])
    print(f"[collect] 进度 {len(done)}/{len(lib)}")


if __name__ == "__main__":
    main()
