### 模型下载与校验

完整模型已下载到：

```text
base_model/STReasoner-8B/
```

4 个权重文件均存在：

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

下载结果：

- `base_model/STReasoner-8B` 总大小约 `16G`。
- 已用 symlink 把本地完整模型接入 `/root/autodl-tmp/cache/huggingface`，使脚本硬编码的 repo id 能离线命中本地权重。

关键结果：

- `model_class`: `Qwen3TSForCausalLM`
- `attn_backend`: `flash_attention_2`
- `model_load_time_sec`: `13.231`
- `generate_success`: `true`
- `parse_success`: `true`
- `actual_new_tokens`: `16`
- GPU 峰值显存约 `15.428 GiB`
- `failure_type`: `no_failure`

