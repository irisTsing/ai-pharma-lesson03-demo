#!/usr/bin/env bash
# 下载并解压 DUD-E 靶点数据（只下载一次，之后走本地缓存）。
# 用法：
#   bash demos/demo_vs/data/download_data.sh            # 默认 DRD3
#   TARGET=akr1c3 bash demos/demo_vs/data/download_data.sh
set -euo pipefail

DEMO_VS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$DEMO_VS_DIR/data"
CACHE_DIR="$DATA_DIR/cache"
TARGET="${TARGET:-drd3}"

case "$TARGET" in
  drd3|akr1c3|egfr) ;;
  *) echo "不支持的靶点：$TARGET（可选 drd3 / akr1c3 / egfr）" >&2; exit 1 ;;
esac

mkdir -p "$CACHE_DIR" "$DATA_DIR/$TARGET"
TARBALL="$CACHE_DIR/${TARGET}.tar.gz"
URL="https://dude.docking.org/targets/${TARGET}/${TARGET}.tar.gz"

if [[ -s "$TARBALL" ]]; then
  echo "[1/3] 已有缓存 $TARBALL（$(du -h "$TARBALL" | cut -f1)），跳过下载"
else
  echo "[1/3] 下载 $URL"
  # DUD-E 会 302 到自己的文件服务，必须 -L 跟随重定向；https 直连
  curl -L --fail --retry 3 --retry-delay 2 -o "$TARBALL.part" "$URL"
  mv "$TARBALL.part" "$TARBALL"
fi

echo "[2/3] 解压到 $DATA_DIR/$TARGET"
tar xzf "$TARBALL" -C "$DATA_DIR"
# tar 内层级是 <target>/，与 $DATA_DIR/$TARGET 一致

echo "[3/3] 补下载 receptor.pdb 与 crystal_ligand.mol2"
# 关键坑：DUD-E 官方 tar 包里这两个文件是指向 docking/ 目录的符号链接，
# 而 tar 包并不包含 docking/，解压后得到的是坏链接。必须从网站单独取。
for f in receptor.pdb crystal_ligand.mol2; do
  if [[ -s "$DATA_DIR/$TARGET/$f" && ! -L "$DATA_DIR/$TARGET/$f" ]]; then
    echo "  已有 $f，跳过"
    continue
  fi
  rm -f "$DATA_DIR/$TARGET/$f"
  curl -L --fail --retry 3 -o "$DATA_DIR/$TARGET/$f" "https://dude.docking.org/targets/${TARGET}/${f}"
  echo "  下载 $f（$(du -h "$DATA_DIR/$TARGET/$f" | cut -f1)）"
done

echo
echo "完成。核对："
wc -l "$DATA_DIR/$TARGET/actives_final.ism" "$DATA_DIR/$TARGET/decoys_final.ism"
grep -c '^ATOM' "$DATA_DIR/$TARGET/receptor.pdb" || true
echo "校验和："
md5sum "$TARBALL" | sed 's/^/  /'
