#!/usr/bin/env bash
# demo 3 一键运行：受体/配体准备 -> Uni-Dock GPU 批量对接 -> vina CPU 对照 -> 打分表与 pose。
# 用法：bash demo_docking/run_demo.sh [--device auto|cuda|cpu] [--n-ligands 10]
# GPU 不可用时自动只跑 vina CPU 版本，并提示原因。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_VS_DIR="$(dirname "$SCRIPT_DIR")"
PY="${VS_PY:-$DEMO_VS_DIR/../venv_chemprop/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DEMO_VS_DIR/../.mplcache}"

if [[ ! -x "$PY" ]]; then
  echo "找不到 Python：$PY（用 VS_PY=... 指定）" >&2
  exit 1
fi

# 数据自检：没下载过就先下载
if [[ ! -s "$DEMO_VS_DIR/data/drd3/actives_final.ism" ]]; then
  echo "[data] 首次运行，先下载 DUD-E DRD3"
  bash "$DEMO_VS_DIR/data/download_data.sh"
  "$PY" "$DEMO_VS_DIR/common/prepare_data.py"
fi
if [[ ! -s "$DEMO_VS_DIR/data/dock_ligands/pdbqt" ]]; then
  echo "[data] 准备配体 3D 构象与 pdbqt"
  "$PY" "$DEMO_VS_DIR/common/ligand_prep.py"
fi

exec "$PY" "$SCRIPT_DIR/run_docking.py" --device auto "$@"
