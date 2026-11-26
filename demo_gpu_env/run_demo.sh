#!/usr/bin/env bash
# demo 0 一键运行：环境自检。
# 用法：bash demo_gpu_env/run_demo.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/check_gpu.sh"
