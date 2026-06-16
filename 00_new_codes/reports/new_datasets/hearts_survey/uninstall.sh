#!/usr/bin/env bash
# HeaRTS 调研资源一键卸载（不影响 STReasoner 复现目录）
set -euo pipefail
DATA_ROOT="/root/autodl-tmp/datasets/HEARTS"
REPORT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 卸载 HEARTS 下载区: $DATA_ROOT =="
rm -rf "$DATA_ROOT"

case "${1:-}" in
  --data-only)
    echo "保留报告与脚本: $REPORT_ROOT"
    ;;
  --with-artifacts)
    echo "删除 artifacts 与 hearts_survey 脚本，保留 Markdown 报告"
    rm -rf "$REPORT_ROOT/artifacts"
    rm -rf "$REPORT_ROOT/hearts_survey"
    ;;
  --all)
    echo "删除整个 new_datasets 目录"
    rm -rf "$REPORT_ROOT"
    ;;
  *)
    echo "用法: $0 [--data-only|--with-artifacts|--all]"
    echo "  --data-only       仅删 /root/autodl-tmp/datasets/HEARTS（推荐）"
    echo "  --with-artifacts  另删 artifacts/ 与 hearts_survey/"
    echo "  --all             删除整个 new_datasets/"
    exit 1
    ;;
esac

if [[ -e "$DATA_ROOT" ]]; then
  echo "ERROR: $DATA_ROOT 仍存在"
  exit 1
fi

if [[ -d /root/autodl-tmp/STReasoner_reproduce/data ]]; then
  echo "OK: STReasoner_reproduce/data 仍在"
else
  echo "WARN: STReasoner data 目录不存在（请人工确认）"
fi
echo "OK: HEARTS 数据已清除"
