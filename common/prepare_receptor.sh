#!/usr/bin/env bash
# 受体准备：DUD-E 提供的 receptor.pdb -> AutoDock Vina / Uni-Dock 可用的 receptor.pdbqt
# 用法：bash demos/demo_vs/common/prepare_receptor.sh [靶点，默认 drd3]
set -euo pipefail

DEMO_VS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-drd3}"
DATA_DIR="$DEMO_VS_DIR/data"
PDB="$DATA_DIR/$TARGET/receptor.pdb"
OUT="$DATA_DIR/$TARGET/receptor.pdbqt"
OBABEL="${OBABEL:-/usr/local/bin/obabel}"

if [[ ! -s "$PDB" ]]; then
  echo "找不到 $PDB，先运行 bash demos/demo_vs/data/download_data.sh" >&2
  exit 1
fi
if [[ -s "$OUT" && "$OUT" -nt "$PDB" && "${FORCE:-0}" != "1" ]]; then
  echo "[receptor] 已有 $OUT（FORCE=1 可重建），跳过"
else
  echo "[receptor] $PDB -> $OUT"
  # -xr: 只保留刚性受体原子（去掉配体/水，按 AutoDock 受体规则）
  # -p 7.4: 按 pH 7.4 加氢（生理 pH）
  # 注意 obabel 2.4 的选项必须写在输入文件之后，写在前面会报 "No input file"
  "$OBABEL" "$PDB" -O "$OUT" -xr -p 7.4 2> "$DATA_DIR/$TARGET/receptor.prep.log"
fi

# 自检：原子数、原子类型、有没有配体/水残留
n_atoms=$(grep -c '^ATOM' "$OUT" || true)
n_hydrogen=$(awk '/^ATOM|^HETATM/{print substr($0, 78, 2)}' "$OUT" | grep -c 'HD' || true)
echo "[receptor] 原子行数 $n_atoms（含极性氢），HD 氢 $n_hydrogen"

# vina 的 pdbqt 每行末尾是 AutoDock 原子类型；出现金属或未知类型要报警
awk '/^ATOM|^HETATM/{print $NF}' "$OUT" | sort | uniq -c | sort -rn | sed 's/^/[receptor]   原子类型 /'

if grep -qE '^(HETATM)' "$OUT"; then
  echo "[receptor] 警告：仍有 HETATM 残留，检查原始 PDB 是否含辅因子或水" >&2
fi
echo "[receptor] 完成：$OUT"
