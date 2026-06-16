# HeaRTS 调研脚本

隔离数据目录：`/root/autodl-tmp/datasets/HEARTS/`

```bash
export HF_HOME=/root/autodl-tmp/datasets/HEARTS/.hf_cache
export https_proxy=http://127.0.0.1:17997 http_proxy=http://127.0.0.1:17997
ART=../artifacts
ROOT=/root/autodl-tmp/datasets/HEARTS/frozen_test_cases

python3 preflight.py --out /root/autodl-tmp/datasets/HEARTS/preflight_result.json
python3 download.py --manifest-out "$ART/download_manifest.json"
python3 inventory.py --root "$ROOT" --code-root /root/autodl-tmp/datasets/HEARTS/code --out "$ART"
python3 inspect_pkl.py --root "$ROOT" --taxonomy "$ART/task_taxonomy.json" \
  --out "$ART/pkl_schema_samples.json" --excerpts-out "$ART/sample_qa_excerpts.json"
python3 render_excerpts_md.py --in "$ART/sample_qa_excerpts.json" --out "$ART/sample_qa_excerpts.md"
```

主报告（v2 深度文字分析，不含统计图）：`01-HeaRTS-HEARTS数据集深度调研.md`

`plot_report.py` 已弃用（v1 产物 `artifacts/fig_*.png` 可手动删除）。

卸载：`bash uninstall.sh --data-only`
