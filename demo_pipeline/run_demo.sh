#!/usr/bin/env bash
# demo 4 一键运行：读教师预计算的 vina 打分 CSV，做 2D 粗筛 top 10% 与全库对接的富集对比。
# 用法：bash demo_pipeline/run_demo.sh [--top-frac 0.10]
# 若结果 CSV 不存在，先尝试用预计算脚本生成（教师课前步骤，CPU 约 30 分钟）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_VS_DIR="$(dirname "$SCRIPT_DIR")"
PY="${VS_PY:-$DEMO_VS_DIR/../venv_chemprop/bin/python}"
CSV="$DEMO_VS_DIR/demo_pipeline/results/full_docking.csv"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DEMO_VS_DIR/../.mplcache}"

if [[ ! -x "$PY" ]]; then
  echo "找不到 Python：$PY（用 VS_PY=... 指定）" >&2
  exit 1
fi
if [[ ! -s "$CSV" ]]; then
  echo "[data] 还没有预计算结果，先跑教师预计算（可后台，约 30 分钟）" >&2
  echo "       $PY demo_pipeline/precompute_docking.py --workers 20 --cpu 3" >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/student_pipeline.py" "$@"
