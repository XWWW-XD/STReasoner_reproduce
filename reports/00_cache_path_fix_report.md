# Cache Path Fix Report

## 新增统一 cache 配置模块

- 新增：`cache_config.py`
- 导入该模块会立即设置并创建缓存目录：
  - `HF_HOME`
  - `HF_HUB_CACHE`
  - `TRANSFORMERS_CACHE`
  - `HF_DATASETS_CACHE`
  - `TORCH_HOME`
- 该模块会覆盖已有环境变量，避免继承到 `C:\Users\HUAWEI\Downloads\working\hf_cache`。
- 该模块提供 `resolve_hub_cache_dir`、`resolve_transformers_cache_dir`、`resolve_datasets_cache_dir`，如果显式传入 C 盘 cache_dir，会直接抛出 `RuntimeError`。
- C 盘拦截会先展开用户目录和环境变量，并拦截 `C:\...`、`C:/...`、`C:...` 形式。

## 修改过的文件

- `cache_config.py`
- `data/filter.py`
- `download_dataset.py`
- `download_model.py`
- `inference/inference_tsmllm_vllm.py`
- `inference/llm_utils.py`
- `initial_model.py`
- `model_merger.py`
- `repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py`
- `repro_kaggle/experiments/scripts/stage1_script/run_experiment1_precision_resource_old.py`
- `repro_kaggle/scripts/00_check_kaggle_env.py`
- `repro_kaggle/scripts/01_setup_kaggle_t4.sh`
- `repro_kaggle/scripts/02_inspect_stbench.py`
- `repro_kaggle/scripts/03_load_streasoner_smoke.py`
- `repro_kaggle/scripts/04_run_one_sttest_sample.py`
- `repro_kaggle/scripts/05_eval_sttest_tiny.py`
- `repro_kaggle/scripts/08_prepare_stage1_exp1_subsets.py`
- `src/EasyR1/scripts/model_merger.py`
- `src/EasyR1/verl/__init__.py`
- `src/EasyR1/verl/utils/dataset.py`
- `src/EasyR1/verl/utils/tokenizer.py`
- `src/EasyR1/verl/workers/fsdp_workers.py`
- `src/api.py`
- `src/cli_demo.py`
- `src/llamafactory/__init__.py`
- `src/llamafactory/chat/sglang_engine.py`
- `src/llamafactory/chat/vllm_engine.py`
- `src/llamafactory/cli.py`
- `src/llamafactory/data/loader.py`
- `src/llamafactory/data/parser.py`
- `src/llamafactory/eval/evaluator.py`
- `src/llamafactory/extras/misc.py`
- `src/llamafactory/launcher.py`
- `src/llamafactory/model/adapter.py`
- `src/llamafactory/model/loader.py`
- `src/llamafactory/model/model_utils/quantization.py`
- `src/llamafactory/model/model_utils/unsloth.py`
- `src/llamafactory/model/model_utils/valuehead.py`
- `src/llamafactory/train/test_utils.py`
- `src/llamafactory/webui/chatter.py`
- `src/llamafactory/webui/common.py`
- `src/llamafactory/webui/components/export.py`
- `src/llamafactory/webui/runner.py`
- `src/train.py`
- `src/webui.py`
- `reports/2026-05-20_cache_path_fix_prompt.md`
- `reports/2026-05-20_cache_path_fix_report.md`

## 原来可能导致写入 C 盘的位置

- `repro_kaggle/scripts/03_load_streasoner_smoke.py` 和 stage1 实验脚本里的 `set_hf_cache_env()` 只设置了 Kaggle 路径，且没有 `HF_HUB_CACHE` / `TORCH_HOME`。
- 多个入口在导入 `transformers`、`datasets`、`huggingface_hub` 或 `torch` 前没有统一设置 cache 环境变量。
- `from_pretrained`、`snapshot_download`、`hf_hub_download`、`load_dataset` 的若干调用没有显式 `cache_dir`，会依赖外部环境或默认用户目录。
- LLaMAFactory / EasyR1 内部的 `model_args.cache_dir` 默认为 `None` 时，会退回库默认缓存；现在改为统一解析到配置模块给出的路径。
- `data/filter.py` 位于子目录，直接以 `python data/filter.py ...` 执行时需要先把项目根目录加入 `sys.path`，否则无法稳定加载统一 cache 配置模块。

## 当前路径策略

Windows / 本地环境：

```text
HF_HOME=D:\hf_cache
HF_HUB_CACHE=D:\hf_cache\hub
TRANSFORMERS_CACHE=D:\hf_cache\transformers
HF_DATASETS_CACHE=D:\hf_cache\datasets
TORCH_HOME=D:\torch_cache
```

Kaggle / Linux 环境：

```text
HF_HOME=/kaggle/working/hf_cache
HF_HUB_CACHE=/kaggle/working/hf_cache/hub
TRANSFORMERS_CACHE=/kaggle/working/hf_cache/transformers
HF_DATASETS_CACHE=/kaggle/working/hf_cache/datasets
TORCH_HOME=/kaggle/working/torch_cache
```

## 不下载模型的验证命令

在项目根目录运行：

```bash
python -c "import os, cache_config; [print(f'{k}={os.environ.get(k)}') for k in ('HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_DATASETS_CACHE','TORCH_HOME')]"
```

本次在 Kaggle/Linux 环境得到：

```text
HF_HOME=/kaggle/working/hf_cache
HF_HUB_CACHE=/kaggle/working/hf_cache/hub
TRANSFORMERS_CACHE=/kaggle/working/hf_cache/transformers
HF_DATASETS_CACHE=/kaggle/working/hf_cache/datasets
TORCH_HOME=/kaggle/working/torch_cache
```

## 需要手动确认的 cache_dir

- 项目仍保留 LLaMAFactory/WebUI/parser 中用户主动指定 `cache_dir` 的能力。
- 如果用户显式指定 C 盘路径，现在会抛错阻止继续运行。
- 如果用户显式指定非 C 盘自定义路径，代码会尊重该路径；如需绝对统一到 D 盘或 Kaggle 路径，请不要额外传 `--cache_dir`，也不要在 WebUI 配置中填自定义 cache_dir。
