#!/usr/bin/env bash
# gpu1 上一次性执行：把本课程要用的 Python 环境装到共享路径，50 名学生直接复用，
# 不要在每个人 home 下重复安装。
#
# 用法（在 gpu1 上，任意目录）：
#   bash demos/demo_vs/setup_gpu_env.sh
#   ENV_PREFIX=/public/AI药学实践/demo_vs/envs/vs bash demos/demo_vs/setup_gpu_env.sh
#   CONDA=/public/software/miniconda3/bin/conda bash demos/demo_vs/setup_gpu_env.sh
#
# 装完会打印三行给学生用的命令（source 环境、设置 VS_PY、自检）。
# 全部步骤可重复执行：已存在的环境不会重装。
set -euo pipefail

# ---- 可改参数 -------------------------------------------------------------
ENV_PREFIX="${ENV_PREFIX:-/public/AI药学实践/demo_vs/envs/vs}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"        # 与 venv_chemprop 的 torch 2.3.1+cu121 对齐；驱动新可用 cu124
PY_VERSION="${PY_VERSION:-3.10}"
# ---------------------------------------------------------------------------

echo "==============================================================="
echo " 课程环境安装（共享路径，一次装好全班用）"
echo "  环境目录：$ENV_PREFIX"
echo "  torch CUDA：$TORCH_CUDA"
echo "==============================================================="

# 1) 机器自检
echo
echo "[1/5] 机器自检"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "  [失败] nvidia-smi 不可用：当前不在 GPU 节点，或驱动没装。请在 gpu1 上执行本脚本。" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
DRIVER_CUDA="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}' | head -1 || true)"
echo "  驱动支持的 CUDA 版本：${DRIVER_CUDA:-未知}"
echo "  提示：Uni-Dock 的 conda 包目前是 CUDA 12.x 构建；驱动版本过低时 unidock 会报 CUDA 初始化错误。"

# 2) conda
echo
echo "[2/5] 查找 conda"
CONDA_BIN="${CONDA:-}"
if [[ -z "$CONDA_BIN" ]]; then
  for cand in \
    "$(command -v conda 2>/dev/null || true)" \
    /public/software/miniconda3/bin/conda \
    /public/software/anaconda3/bin/conda \
    /opt/conda/bin/conda \
    "$HOME/miniconda3/bin/conda"; do
    if [[ -n "$cand" && -x "$cand" ]]; then CONDA_BIN="$cand"; break; fi
  done
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "  [失败] 找不到 conda。用 CONDA=/path/to/conda 指定。" >&2
  exit 1
fi
echo "  使用 conda：$CONDA_BIN"
# 共享目录里建环境，conda 的包缓存也放到共享路径，避免写满 home
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ENV_PREFIX/.conda_pkgs}"
mkdir -p "$CONDA_PKGS_DIRS" "$(dirname "$ENV_PREFIX")"

# 3) Python 环境
echo
echo "[3/5] 创建 Python 环境（$PY_VERSION）"
if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  echo "  已存在，跳过创建"
else
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge "python=$PY_VERSION" pip
fi
PY="$ENV_PREFIX/bin/python"

# 4) Python 包（torch CUDA 版 + 化学信息学栈）
echo
echo "[4/5] 安装 torch（$TORCH_CUDA）与 RDKit / sklearn / matplotlib / pandas / numpy"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" "torch==2.3.1" 2>&1 | tail -2
"$PY" -m pip install "numpy==1.26.4" "scikit-learn==1.5.0" "pandas==2.2.2" "matplotlib==3.9.0" "rdkit==2025.3.3" 2>&1 | tail -2

# 5) Uni-Dock
echo
echo "[5/5] 安装 Uni-Dock（GPU 对接，Apache 2.0）"
if "$CONDA_BIN" list -p "$ENV_PREFIX" 2>/dev/null | grep -q '^unidock '; then
  echo "  已安装，跳过"
else
  # Uni-Dock 依赖较新的 CUDA 运行时；装不上时不影响 demo 1/2，demo 3 可先用 vina CPU 回退
  "$CONDA_BIN" install -y -p "$ENV_PREFIX" -c conda-forge unidock 2>&1 | tail -3 || {
    echo "  [警告] Uni-Dock 安装失败。demo 3 先用 --device cpu 跑 vina；"
    echo "         也可以试 pip install unidock，或在 gpu1 上确认驱动 CUDA 版本后重试。"
  }
fi

# ---- 自检 -----------------------------------------------------------------
echo
echo "==============================================================="
echo " 自检"
echo "==============================================================="
"$PY" - <<'PYEOF'
import sys, torch, rdkit, sklearn, pandas, numpy, matplotlib
print(f"  python      {sys.version.split()[0]}")
print(f"  torch       {torch.__version__}  (CUDA build {torch.version.cuda})")
print(f"  rdkit       {rdkit.__version__}")
print(f"  sklearn     {sklearn.__version__}")
print(f"  pandas      {pandas.__version__}")
print(f"  numpy       {numpy.__version__}")
print(f"  cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU         {p.name}  {p.total_memory/1024**3:.1f} GB  capability {p.major}.{p.minor}")
    x = torch.randn(2048, 2048, device="cuda"); _ = x @ x.T; torch.cuda.synchronize()
    print("  GPU 矩阵乘自检通过")
else:
    print("  [警告] torch 看不到 GPU：demo 1/2/3 会回退 CPU，先确认驱动与 CUDA_VISIBLE_DEVICES")
PYEOF

UNIDOCK_BIN="$ENV_PREFIX/bin/unidock"
if [[ -x "$UNIDOCK_BIN" ]]; then
  echo "  unidock     $("$UNIDOCK_BIN" --version 2>&1 | head -1)"
else
  echo "  unidock     未安装（demo 3 用 CPU 回退）"
fi

echo
echo "==============================================================="
echo " 给学生的三条命令"
echo "==============================================================="
echo "  export PATH=$ENV_PREFIX/bin:\$PATH"
echo "  bash demos/demo_vs/demo_gpu_env/check_gpu.sh"
echo
echo "跑 demo 时用 VS_PY 指到共享环境的 python："
echo "  export VS_PY=$ENV_PREFIX/bin/python"
echo
echo "可选：把这两行写进 /public 共享目录里的 env.sh，全班组会一次 source 即可。"
echo "完成。"
