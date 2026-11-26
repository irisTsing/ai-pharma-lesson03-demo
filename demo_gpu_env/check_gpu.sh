#!/usr/bin/env bash
# 学生上 gpu1 后的第一件事：确认自己能看到哪张卡、torch 能不能用。
# 用法：
#   bash demo_gpu_env/check_gpu.sh            # 检查全部 8 张卡
##   CUDA_VISIBLE_DEVICES=3 bash check_gpu.sh  # 手动指定卡
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_VS_DIR="$(dirname "$SCRIPT_DIR")"
PY="${VS_PY:-$DEMO_VS_DIR/../venv_chemprop/bin/python}"
STATUS=0

echo "==============================================================="
echo " GPU 环境自检"
echo "==============================================================="
echo "主机：$(hostname)   用户：$(whoami)   时间：$(date '+%F %T')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置，表示可见全部卡>}"

# 1) 驱动与显卡列表
echo
echo "[1/3] nvidia-smi 驱动与显卡"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || STATUS=1
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "  [失败] nvidia-smi 命令存在但执行失败：驱动未加载，或当前不在 GPU 节点。"
    STATUS=1
  fi
else
  echo "  [失败] 找不到 nvidia-smi：当前机器不是 GPU 节点，或驱动没装。"
  STATUS=1
fi

# 2) 显存占用与谁在跑（多人共用一张卡时，先看还剩多少）
echo
echo "[2/3] 显存占用（used / total）"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
  echo
  echo "  当前占用进程（前 10 行）："
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -10 | sed 's/^/    /' || true
else
  echo "  [跳过] 没有可用驱动"
fi

# 3) torch CUDA 自检
echo
echo "[3/3] PyTorch CUDA 自检（$PY）"
if [[ ! -x "$PY" ]]; then
  echo "  [失败] 找不到 Python：$PY"
  echo "  用 VS_PY=/path/to/python 指定，或先执行 source setup_gpu_env.sh 里给出的激活命令。"
  STATUS=1
else
  "$PY" - <<'PYEOF' || STATUS=1
import os, sys
try:
    import torch
except Exception as exc:
    print(f"  [失败] 导入 torch 失败：{exc}")
    print("  对策：确认用共享环境里的 python，而不是系统 python。")
    sys.exit(1)
print(f"  torch 版本：{torch.__version__}   编译 CUDA：{torch.version.cuda}")
print(f"  cuda.is_available()：{torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  [失败] torch 看不到 GPU。检查：")
    print("    1) 是否在 gpu1 上（hostname）；2) nvidia-smi 是否正常；")
    print("    3) 装的 torch 是不是 CPU 版；4) CUDA_VISIBLE_DEVICES 是否指向一张存在的卡。")
    sys.exit(1)
n = torch.cuda.device_count()
print(f"  可见 GPU 数量：{n}")
for i in range(n):
    prop = torch.cuda.get_device_properties(i)
    print(f"    cuda:{i}  {prop.name}  显存 {prop.total_memory/1024**3:.1f} GB  capability {prop.major}.{prop.minor}")
# 做一次小矩阵乘，确认真的能算
x = torch.randn(1024, 1024, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print(f"  矩阵乘自检通过，结果范数 {float(y.norm()):.3f}")
print(f"  当前显存占用 {torch.cuda.memory_allocated()/1024**2:.1f} MB")
PYEOF
fi

echo
if [[ $STATUS -eq 0 ]]; then
  echo "结论：GPU 环境正常，可以开始跑 demo 1/2/3。"
else
  echo "结论：GPU 环境有问题。先用 --device cpu 跑教学 demo，同时把上面的报错发给助教。"
fi
exit $STATUS
