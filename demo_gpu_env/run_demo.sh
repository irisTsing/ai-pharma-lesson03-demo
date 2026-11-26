#!/usr/bin/env bash
# demo 0 一键运行：按学号分卡 + 环境自检。
# 用法：bash demo_gpu_env/run_demo.sh [学号]
# 不带学号时只做环境检查，不设置分卡。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDENT_ID="${1:-}"

if [[ -n "$STUDENT_ID" ]]; then
  eval "$(bash "$SCRIPT_DIR/gpu_assign.sh" "$STUDENT_ID" --export)"
  echo "已设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES（学号 $STUDENT_ID，第 $VS_GROUP 组）"
fi

bash "$SCRIPT_DIR/check_gpu.sh"
