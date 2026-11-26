#!/usr/bin/env bash
# demo 1 一键运行：ECFP4 + Tanimoto 相似性筛选，输出 top-k 与富集指标。
# 用法：bash demo_similarity/run_demo.sh [--device auto|cuda|cpu] [--topk 20]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_VS_DIR="$(dirname "$SCRIPT_DIR")"
PY="${VS_PY:-$DEMO_VS_DIR/../venv_chemprop/bin/python}"

# 课堂 50 人共用节点：限制 BLAS / OMP 线程，避免每个学生开 64 线程把机器拖死
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DEMO_VS_DIR/../.mplcache}"

if [[ ! -x "$PY" ]]; then
  echo "找不到 Python：$PY" >&2
  echo "用 VS_PY=/path/to/python 指定，或先运行 setup_gpu_env.sh（gpu1）。" >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/demo_similarity.py" --device auto "$@"
