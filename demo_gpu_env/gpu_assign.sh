#!/usr/bin/env bash
# 按学号分卡：GPU_ID = 学号 % 8，组号 = GPU_ID + 1。
# 8 张 RTX 3090，50 名学生时每卡 6～7 人。
#
# 用法：
#   bash demo_gpu_env/gpu_assign.sh 2023012345
#   eval "$(bash demo_gpu_env/gpu_assign.sh 2023012345 --export)"
#   bash demo_gpu_env/gpu_assign.sh --table    # 打印分组表，贴到教室屏幕上
set -euo pipefail

N_GPU="${N_GPU:-8}"

print_table() {
  echo "学号尾号规则：GPU_ID = 学号 % ${N_GPU}（0～$((N_GPU - 1))）"
  echo
  printf '%-8s %-16s %-8s %s\n' "GPU_ID" "CUDA_VISIBLE_DEVICES" "组号" "组内人数（50 人时）"
  for ((g = 0; g < N_GPU; g++)); do
    n_students=$(( (50 + N_GPU - 1 - g) / N_GPU ))
    printf '%-8s %-16s %-8s %s\n' "$g" "$g" "$((g + 1))" "$n_students"
  done
  echo
  echo "每张卡 6～7 人，Uni-Dock 每次调用必须加 --max_gpu_memory 2500（单位 MB）。"
}

if [[ "${1:-}" == "--table" || "${1:-}" == "-t" ]]; then
  print_table
  exit 0
fi

if [[ $# -lt 1 || ! "${1:-}" =~ ^[0-9]+$ ]]; then
  echo "用法：bash demo_gpu_env/gpu_assign.sh <学号，纯数字> [--export]" >&2
  echo "      bash demo_gpu_env/gpu_assign.sh --table" >&2
  exit 1
fi

STUDENT_ID="$1"
MODE="${2:-}"
GPU_ID=$(( STUDENT_ID % N_GPU ))
GROUP=$(( GPU_ID + 1 ))

if [[ "$MODE" == "--export" ]]; then
  # 给 eval 用，只输出 shell 赋值
  echo "export CUDA_VISIBLE_DEVICES=$GPU_ID"
  echo "export VS_GPU_ID=$GPU_ID"
  echo "export VS_GROUP=$GROUP"
  exit 0
fi

echo "学号：$STUDENT_ID"
echo "GPU_ID = $STUDENT_ID % $N_GPU = $GPU_ID"
echo "组号：$GROUP"
echo
echo "执行下面这一行（或者直接 eval 本脚本的 --export 输出）："
echo "  export CUDA_VISIBLE_DEVICES=$GPU_ID"
echo
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "这张卡当前状态："
  nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /' || true
  echo "这张卡上正在跑的同学进程："
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
fi
echo
echo "分完卡先跑：bash demo_gpu_env/check_gpu.sh"
