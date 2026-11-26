#!/usr/bin/env bash
# demo 2 一键运行：ECFP4 + 随机森林(CPU) / 多层感知机(GPU)，与 demo 1 对照。
# 用法：bash demo_ml_screen/run_demo.sh [--device auto|cuda|cpu] [--epochs 30]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_VS_DIR="$(dirname "$SCRIPT_DIR")"
PY="${VS_PY:-$DEMO_VS_DIR/../venv_chemprop/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DEMO_VS_DIR/../.mplcache}"

if [[ ! -x "$PY" ]]; then
  echo "找不到 Python：$PY（用 VS_PY=... 指定）" >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/demo_ml_screen.py" --device auto "$@"
