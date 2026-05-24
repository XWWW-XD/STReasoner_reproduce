# Prompt: cache path fix

请你只在当前项目目录内修改代码，先修复 Hugging Face / transformers / datasets / torch 缓存路径问题，避免后续模型缓存继续写入 C 盘。不要跑模型，不要改实验逻辑、评测逻辑、parser 逻辑。

背景事故：
我在 VS Code 中以为自己连接 Kaggle 远程运行 STReasoner 实验，但本地 C 盘空间从 20GB 左右快速下降到 0.5GB 左右。排查发现模型缓存实际写入了本地路径：

`C:\Users\HUAWEI\Downloads\working\hf_cache\transformers\models--Time-HD-Anonymous--STReasoner...`

其中出现多个约 1.52GB 的模型分片文件。这说明当前实验脚本或运行环境中，Hugging Face cache 仍然可能指向本地 C 盘，而不是 D 盘或 Kaggle 远程路径。

目标：
统一项目中的缓存路径设置，确保后续不会继续向 C 盘写入大模型缓存。

请搜索项目中所有与缓存相关的位置，包括：
- HF_HOME
- HF_HUB_CACHE
- HF_DATASETS_CACHE
- TRANSFORMERS_CACHE
- TORCH_HOME
- cache_dir
- from_pretrained
- snapshot_download
- load_dataset
- hf_cache

请按运行环境自动选择缓存目录：

Windows / 本地环境：
- HF_HOME = D:\hf_cache
- HF_HUB_CACHE = D:\hf_cache\hub
- TRANSFORMERS_CACHE = D:\hf_cache\transformers
- HF_DATASETS_CACHE = D:\hf_cache\datasets
- TORCH_HOME = D:\torch_cache

Kaggle / Linux 环境：
- HF_HOME = /kaggle/working/hf_cache
- HF_HUB_CACHE = /kaggle/working/hf_cache/hub
- TRANSFORMERS_CACHE = /kaggle/working/hf_cache/transformers
- HF_DATASETS_CACHE = /kaggle/working/hf_cache/datasets
- TORCH_HOME = /kaggle/working/torch_cache

实现要求：
1. 优先新增一个统一缓存配置模块，例如：
   `repro_kaggle/experiments/scripts/stage1_script/cache_config.py`
   或项目中你认为更合适的位置。
2. 在所有实验入口脚本的最开头导入这个模块，确保缓存环境变量在 import transformers / datasets / huggingface_hub / torch 之前设置。
3. 如果 from_pretrained、snapshot_download、load_dataset 里已有 cache_dir 参数，请统一改为读取 cache_config.py 中的路径，不要写死 C 盘或 /kaggle 路径。
4. 不允许再把任何模型缓存、数据集缓存、torch cache 写到：
   `C:\Users\HUAWEI\Downloads\working\hf_cache`
   或任何 C 盘路径。
5. 不要删除任何数据文件，不要清理 D 盘，不要移动已有模型文件。
6. 不要运行会下载大模型的命令，只做静态修改和小型验证。
7. 修改后请提供一个不下载模型的小验证命令，只打印这些环境变量：
   HF_HOME
   HF_HUB_CACHE
   TRANSFORMERS_CACHE
   HF_DATASETS_CACHE
   TORCH_HOME
8. 修改完成后请在 reports 中保存这次的提示词和这次的报告，可以包括以下内容：
   - 修改了哪些文件；
   - 新增的统一 cache 配置模块在哪里；
   - 原来哪些地方可能导致缓存写到 C 盘；
   - Windows 本地环境现在会写到哪里；
   - Kaggle / Linux 环境现在会写到哪里；
   - 如何用不下载模型的小命令验证环境变量；
   - 是否仍存在需要我手动确认的 cache_dir。
