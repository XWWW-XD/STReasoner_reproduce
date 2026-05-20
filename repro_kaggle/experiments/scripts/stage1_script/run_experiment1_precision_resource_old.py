#!/usr/bin/env python3
"""实验一：不同精度推理资源测试。

脚本放在实验一自己的结果目录下，不放入全局 repro_kaggle/scripts。

默认只执行 prepare：检查固定样本、写证据和四组配置空表，等待人工确认。
确认后可用 run-all 或 run-config 执行模型实验。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cache_config import TRANSFORMERS_CACHE_PATH, apply_cache_config

apply_cache_config()

RESULT_ROOT = PROJECT_ROOT / "repro_kaggle/experiments/stage1_results/experiment1_precision_resource"
DOC_ROOT = PROJECT_ROOT / "repro_kaggle/experiments/stage1_docs"
SCRIPT_PATH = PROJECT_ROOT / "repro_kaggle/experiments/scripts/stage1_script/run_experiment1_precision_resource.py"

MODEL_NAME = "Time-HD-Anonymous/STReasoner-8B"
MAX_NEW_TOKENS = 512
PATCH_SIZE = 8

MAIN_DATA = (
    PROJECT_ROOT
    / "repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl"
)
PAPER_DATA = (
    PROJECT_ROOT / "repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/paper_cases/paper_cases_matched.jsonl"
)
STRESS_DATA = (
    PROJECT_ROOT / "repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/stress_case/stress_longest_input_1.jsonl"
)

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
CHOICE_RE = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)")


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    label: str
    precision: str
    cuda_visible_devices: str
    device_map_kind: str


CONFIGS: dict[str, ConfigSpec] = {
    "4bit_single": ConfigSpec("4bit_single", "4bit单卡", "4bit", "0", "single"),
    "8bit_single": ConfigSpec("8bit_single", "8bit单卡", "8bit", "0", "single"),
    "fp16_single": ConfigSpec("fp16_single", "fp16单卡", "fp16", "0", "single"),
    "fp16_dual": ConfigSpec("fp16_dual", "fp16双卡", "fp16", "0,1", "balanced"),
}


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._fh.write(message + "\n")
        self._fh.flush()

    def exception(self, title: str, exc: BaseException) -> None:
        self.log(f"[ERROR] {title}: {exc.__class__.__name__}: {exc}")
        self.log(traceback.format_exc())

    def close(self) -> None:
        self._fh.close()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def sample_task(sample: dict[str, Any]) -> str:
    return str(sample.get("task") or sample.get("category") or "unknown")


def validate_sample_file(path: Path, expected_count: int, group_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{group_name} 样本文件不存在: {path}")
    rows = load_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(f"{group_name} 样本数量应为 {expected_count}，实际为 {len(rows)}: {path}")
    for index, sample in enumerate(rows):
        for key in ("input", "timeseries", "output"):
            if key not in sample:
                raise ValueError(f"{group_name} 第 {index} 条缺少字段 {key}")
        prompt = sample["input"]
        timeseries = sample["timeseries"]
        if not isinstance(prompt, str) or not isinstance(timeseries, list):
            raise TypeError(f"{group_name} 第 {index} 条 input/timeseries 类型不正确")
        placeholder_count = prompt.count("<ts><ts/>")
        if placeholder_count != len(timeseries):
            raise ValueError(
                f"{group_name} 第 {index} 条 <ts><ts/> 数量和 timeseries 节点数不一致: "
                f"{placeholder_count} vs {len(timeseries)}"
            )
    return rows


def validate_inputs() -> dict[str, Any]:
    main_rows = validate_sample_file(MAIN_DATA, 20, "主测试")
    paper_rows = validate_sample_file(PAPER_DATA, 4, "论文样例")
    stress_rows = validate_sample_file(STRESS_DATA, 1, "压力测试")
    counts = Counter(sample_task(row) for row in main_rows)
    expected = {"forecasting": 5, "entity": 5, "etiological": 5, "correlation": 5}
    if dict(counts) != expected:
        raise ValueError(f"主测试四类任务数量不符合预期: {dict(counts)}")
    return {
        "main_count": len(main_rows),
        "paper_count": len(paper_rows),
        "stress_count": len(stress_rows),
        "main_task_counts": dict(counts),
    }


def evidence_section() -> str:
    return """## 证据查找

### 作者代码中的精度设置

- 训练脚本使用 fp16：
  - `scripts/qwen3-8b/train_stage1.sh:24`：`--fp16`
  - `scripts/qwen3-8b/train_stage2.sh:24`：`--fp16`
- 推理脚本中的 dtype：
  - `inference/llm_utils.py:97-98`：普通 vLLM worker 使用 `LLM(..., dtype='half')`，对应 fp16/half。
  - `inference/llm_utils.py:142` 之后的 `worker_vllm_ts` 未显式设置 dtype；实际 dtype 需要通过本实验记录加载后模型参数 dtype、量化配置和日志确认。

### 作者代码中的 max_new_tokens / max_tokens 设置

- 作者 vLLM 推理使用 `max_tokens=512`：
  - `inference/inference_tsmllm_vllm.py:64-68`：`SamplingParams(max_tokens=512, temperature=0.2)`
- 本实验使用 Hugging Face `generate`，对应参数记录为 `max_new_tokens=512`。
- Hugging Face 的 `max_new_tokens` 表示最多生成的新 token 数，不包含 prompt tokens。
"""


def empty_summary_table() -> str:
    return """## 实验记录表（空表，待确认后填数值）

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|       样本       | 主测试 20 + 论文样例 4 + 压力测试 1 |
|   batch size   |                1                |
| max_new_tokens |               512               |

|           | dtype                        | 4bit单卡 | 8bit单卡 | fp16单卡 | fp16双卡 |
| --------- | ---------------------------- | -------: | -------: | -------: | ------- |
| 配置证据      | 加载方式                         | 单卡 / 双卡 / offload | 单卡 / 双卡 / offload | 单卡 / 双卡 / offload | 单卡 / 双卡 / offload |
|           | device_map                   | 单卡 / auto / balanced | 单卡 / auto / balanced | 单卡 / auto / balanced | 单卡 / auto / balanced |
|           | 实际模型分布                       | 全在 cuda:0 / 分布到两卡 / 有 CPU | 全在 cuda:0 / 分布到两卡 / 有 CPU | 全在 cuda:0 / 分布到两卡 / 有 CPU | 全在 cuda:0 / 分布到两卡 / 有 CPU |
|           | is_cpu_offload               | 无 / CPU / disk | 无 / CPU / disk | 无 / CPU / disk | 无 / CPU / disk |
|           | use_cache                    | 默认通常是 True | 默认通常是 True | 默认通常是 True | 默认通常是 True |
| 可运行证据     | input tokens（平均值）            | 实际值 | 实际值 | 实际值 | 实际值 |
|           | actual new tokens（平均值）       | 实际值 | 实际值 | 实际值 | 实际值 |
|           | load 成功率                     | 成功/失败 | 成功/失败 | 成功/失败 | 成功/失败 |
|           | generate 成功率                 | 成功/失败 | 成功/失败 | 成功/失败 | 成功/失败 |
| 资源与速度     | GPU 总显存（若是双卡则分别记录）           |  |  |  |  |
|           | load 后显存（若是双卡则分别记录）          | X GB | X GB | X GB | X GB |
|           | generate 峰值显存（若是双卡则分别记录）     | X GB | X GB | X GB | X GB |
|           | 平均延迟与最高延迟                    | X 秒 | X 秒 | X 秒 | X 秒 |
|           | tokens/s                     | X | X | X | X |
|           | decode 成功率                   | 正常/异常 | 正常/异常 | 正常/异常 | 正常/异常 |
|           | parse 成功率                    | 成功/失败 | 成功/失败 | 成功/失败 | 成功/失败 |
|           | 平均正确率（对比失败也算错误）              |  |  |  |  |
| 失败阶段、失败原因 | 失败阶段、详细失败原因；若有输出，输出是否正确（T/F） |  |  |  |  |
"""


def write_prepare_docs() -> None:
    validation = validate_inputs()
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    summary = f"""# 实验一：不同精度推理资源测试

> 当前状态：已完成证据查找和表体准备，尚未运行模型。请先确认下方“四组配置总表”表体是否正确，再填入数值。

## 样本与目录

- 主测试样例：`{rel(MAIN_DATA)}`，共 {validation['main_count']} 条，四类任务各 5 条。
- 论文样例：`{rel(PAPER_DATA)}`，共 {validation['paper_count']} 条。
- 压力测试样例：`{rel(STRESS_DATA)}`，共 {validation['stress_count']} 条。
- 运行脚本：`{rel(SCRIPT_PATH)}`
- 机器可读结果目录：`{rel(RESULT_ROOT)}`

{evidence_section()}

{empty_summary_table()}

## 说明

- 本实验名称统一为“实验一”；tiny20、paper cases、stress 只作为三组测试样例，不作为实验名。
- 后续正式成功率统计三组样例合计 25 条：主测试 20 条 + 论文样例 4 条 + 压力测试 1 条。
- 报告仍会分别保留主测试、论文样例和压力测试的分组结果，方便定位问题。
- 失败原因会按资源瓶颈、输入/生成瓶颈、输出与评测瓶颈三类总结；确定的原因尽量细写，不确定时只写粗略阶段和错误信息。
"""
    (DOC_ROOT / "experiment_summary.md").write_text(summary, encoding="utf-8")

    for config_name, spec in CONFIGS.items():
        detail = f"""# 实验一：{spec.label}

> 当前状态：等待总表表体确认，尚未运行该配置。

## 固定配置

- 模型：`STReasoner_8B` / `{MODEL_NAME}`
- 样本：主测试 20 条 + 论文样例 4 条 + 压力测试 1 条
- batch size：1
- max_new_tokens：512
- do_sample：False
- 配置名：`{config_name}`

## 待记录内容

- 主测试样例结果
- 论文样例结果
- 压力测试结果
- 三组样例合计的正式成功率
- 资源、速度、输出格式、parse 和正确率
- 失败阶段与瓶颈类型
"""
        (DOC_ROOT / f"experiment1_{config_name}.md").write_text(detail, encoding="utf-8")


def parse_choice(text: Any) -> tuple[str | None, str | None]:
    if text is None:
        return None, "empty"
    value = str(text).strip()
    if not value:
        return None, "empty"
    tag = ANSWER_TAG_RE.search(value)
    if tag:
        content = tag.group(1).strip()
        if content in {"A", "B", "C", "D"}:
            return content, None
        match = CHOICE_RE.search(content)
        if match:
            return match.group(1), None
        return None, f"answer_tag_not_choice: {content[:100]}"
    if value in {"A", "B", "C", "D"}:
        return value, None
    match = CHOICE_RE.search(value)
    if match:
        return match.group(1), None
    return None, "no_answer_tag_or_choice"


def parse_forecasting(text: Any) -> tuple[Any | None, str | None]:
    if text is None:
        return None, "empty"
    value = str(text)
    tag = ANSWER_TAG_RE.search(value)
    candidate = tag.group(1).strip() if tag else value.strip()
    try:
        parsed = json.loads(candidate)
    except Exception:
        match = re.search(r"\[[^\]]+\]", candidate)
        if not match:
            return None, "no_json_array"
        try:
            parsed = json.loads(match.group(0))
        except Exception as exc:
            return None, f"json_array_parse_error: {exc}"
    if isinstance(parsed, list) and all(isinstance(item, (int, float)) for item in parsed):
        return [float(item) for item in parsed], None
    return None, "forecast_not_numeric_list"


def parse_answer(text: Any, task: str) -> tuple[Any | None, str | None]:
    if task == "forecasting":
        return parse_forecasting(text)
    return parse_choice(text)


def is_correct_prediction(pred: Any, gold: Any, task: str) -> bool:
    if task == "forecasting":
        if not isinstance(pred, list) or not isinstance(gold, list) or len(pred) != len(gold):
            return False
        return all(abs(float(a) - float(b)) <= 1e-3 for a, b in zip(pred, gold))
    return pred == gold


def set_hf_cache_env() -> None:
    apply_cache_config()


def import_repro_loader() -> Any:
    import importlib.util

    loader_path = PROJECT_ROOT / "repro_kaggle/scripts/03_load_streasoner_smoke.py"
    spec = importlib.util.spec_from_file_location("experiment1_repro_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 {loader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment1_repro_loader"] = module
    spec.loader.exec_module(module)
    return module


def import_eval_patch_module() -> Any:
    import importlib.util

    patch_path = PROJECT_ROOT / "repro_kaggle/scripts/05_eval_sttest_tiny.py"
    spec = importlib.util.spec_from_file_location("experiment1_eval_sttest_tiny_patch", patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 {patch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment1_eval_sttest_tiny_patch"] = module
    spec.loader.exec_module(module)
    return module


def apply_timeseries_merge_patch(model: Any, logger: TeeLogger) -> bool:
    patch_module = import_eval_patch_module()
    patch_module.patch_timeseries_merge_device(model, logger)
    applied = bool(getattr(model, "_repro_kaggle_merge_patch", False))
    logger.log(f"MERGE_PATCH_APPLIED={applied}")
    return applied


def gpu_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"visible_gpu_count": 0, "gpu_names": [], "gpu_total_gib": {}}
    names = []
    totals = {}
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        names.append(props.name)
        totals[f"gpu{idx}"] = round(props.total_memory / 1024**3, 3)
    return {"visible_gpu_count": torch.cuda.device_count(), "gpu_names": names, "gpu_total_gib": totals}


def gpu_memory_snapshot() -> dict[str, dict[str, float]]:
    import torch

    if not torch.cuda.is_available():
        return {}
    snapshot = {}
    for idx in range(torch.cuda.device_count()):
        snapshot[f"gpu{idx}"] = {
            "allocated_gib": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(idx) / 1024**3, 3),
        }
    return snapshot


def gpu_peak_snapshot() -> dict[str, dict[str, float]]:
    import torch

    if not torch.cuda.is_available():
        return {}
    snapshot = {}
    for idx in range(torch.cuda.device_count()):
        snapshot[f"gpu{idx}"] = {
            "max_allocated_gib": round(torch.cuda.max_memory_allocated(idx) / 1024**3, 3),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved(idx) / 1024**3, 3),
        }
    return snapshot


def reset_gpu_peak() -> None:
    import torch

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(idx)
            except RuntimeError:
                # Some CUDA builds raise "Invalid device argument" before a device has
                # active allocator stats. Peak memory can still be read later; do not
                # turn measurement reset into an experiment failure.
                continue


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_map_for(spec: ConfigSpec) -> Any:
    if spec.device_map_kind == "single":
        return {"": 0}
    if spec.device_map_kind == "balanced":
        return "balanced"
    raise ValueError(f"未知 device_map_kind: {spec.device_map_kind}")


def quantization_config_for(spec: ConfigSpec) -> Any | None:
    import torch
    from transformers import BitsAndBytesConfig

    if spec.precision == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if spec.precision == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(spec: ConfigSpec, logger: TeeLogger) -> tuple[Any, Any, Any, float, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM

    repro_loader = import_repro_loader()
    started = time.perf_counter()
    processor, tokenizer = repro_loader.load_processor_and_tokenizer(MODEL_NAME, logger)
    config = repro_loader.load_config(MODEL_NAME, "sdpa", logger)
    qconfig = quantization_config_for(spec)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map_for(spec),
        "torch_dtype": torch.float16,
        "config": config,
        "cache_dir": TRANSFORMERS_CACHE_PATH,
    }
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig
    logger.log(f"加载配置: precision={spec.precision}, device_map={kwargs['device_map']}, quantization={qconfig}")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    except Exception as exc:
        logger.exception("AutoModelForCausalLM 加载失败，尝试 AutoModel", exc)
        model = AutoModel.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    load_time = time.perf_counter() - started
    info = {
        "actual_device_map": json_safe(getattr(model, "hf_device_map", None)),
        "use_cache": getattr(getattr(model, "config", None), "use_cache", None),
        "first_parameter_dtype": str(next(model.parameters()).dtype),
        "load_after_memory": gpu_memory_snapshot(),
    }
    return model, processor, tokenizer, load_time, info


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


def first_model_device(model: Any) -> Any:
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def decode_prediction(outputs: Any, inputs: Any, tokenizer: Any, processor: Any) -> tuple[str | None, bool, str | None, int | None]:
    decoder = tokenizer if tokenizer is not None else processor
    if decoder is None or not hasattr(decoder, "decode"):
        return None, False, "无可用 decoder", None
    try:
        generated_ids = outputs[0]
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        actual_new_tokens = None
        if input_ids is not None and hasattr(input_ids, "shape"):
            actual_new_tokens = int(generated_ids.shape[-1] - input_ids.shape[-1])
            generated_ids = generated_ids[input_ids.shape[-1] :]
        text = decoder.decode(generated_ids, skip_special_tokens=True)
        return text, True, None, actual_new_tokens
    except Exception as exc:
        return None, False, f"{exc.__class__.__name__}: {exc}", None


def estimate_ts_tokens(timeseries: Any) -> int:
    if not isinstance(timeseries, list):
        return 0
    total = 0
    for series in timeseries:
        if isinstance(series, list):
            total += math.ceil(len(series) / PATCH_SIZE)
    return total


def build_inputs(processor: Any, tokenizer: Any, sample: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    prompt = sample["input"]
    timeseries = sample["timeseries"]
    inputs = processor(text=prompt, timeseries=timeseries, return_tensors="pt")
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    tokenizer_prompt_tokens = None
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        tokenizer_prompt_tokens = len(tokenizer.encode(prompt))
    processor_input_tokens = int(input_ids.shape[-1]) if input_ids is not None and hasattr(input_ids, "shape") else None
    ts_tokens = estimate_ts_tokens(timeseries)
    return inputs, {
        "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
        "processor_input_ids_length": processor_input_tokens,
        "timeseries_patch_tokens_estimate": ts_tokens,
        "input_tokens_metric": (processor_input_tokens or tokenizer_prompt_tokens or 0) + ts_tokens,
    }


def classify_bottleneck(stage: str, message: str | None, actual_new_tokens: int | None) -> str | None:
    if stage in {"model_loading", "generate"}:
        msg = (message or "").lower()
        if "size of tensor" in msg or "must match the size" in msg or "non-singleton dimension" in msg:
            return "输入/生成瓶颈"
        if "out of memory" in msg or "cuda oom" in msg or "offload" in msg or "device" in msg:
            return "资源瓶颈"
        if actual_new_tokens is not None and actual_new_tokens < MAX_NEW_TOKENS:
            return "输入/生成瓶颈"
    if stage in {"decode", "parse", "score"}:
        return "输出与评测瓶颈"
    if stage != "none":
        return "资源瓶颈"
    return None


def concise_failure_reason(stage: str, message: str | None, actual_new_tokens: int | None) -> str | None:
    """写确定的失败原因；不确定时只写粗略分类。"""

    if stage == "none":
        if actual_new_tokens is not None and actual_new_tokens < MAX_NEW_TOKENS:
            return f"生成提前结束：actual_new_tokens={actual_new_tokens}，小于 max_new_tokens={MAX_NEW_TOKENS}；具体原因可能是 EOS 或模型自然停止。"
        return None

    msg = (message or "").strip()
    lower = msg.lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return f"显存不足导致失败：{msg}"
    if "expected all tensors to be on the same device" in lower or "device mismatch" in lower:
        return f"设备不一致导致失败：{msg}"
    if "offload" in lower:
        return f"发生 offload 相关失败：{msg}"
    if "size of tensor" in lower or "must match the size" in lower or "non-singleton dimension" in lower:
        return f"输入与 time-series 特征合并时张量长度不匹配，generate 失败：{msg}"
    if stage == "decode":
        return f"decode 阶段失败：{msg or '无法从生成 ids 解码文本。'}"
    if stage == "parse":
        return f"parse 阶段失败：{msg or '输出未能解析为预期答案格式。'}"
    if stage == "generate":
        return f"generate 阶段失败，具体原因见错误信息：{msg}" if msg else "generate 阶段失败，具体原因不明。"
    if stage == "model_loading":
        return f"模型加载失败，具体原因见错误信息：{msg}" if msg else "模型加载失败，具体原因不明。"
    return f"{stage} 阶段失败：{msg}" if msg else f"{stage} 阶段失败，具体原因不明。"


def base_record(sample: dict[str, Any], group: str) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "sample_group": group,
        "task": sample_task(sample),
        "source_file": sample.get("source_file"),
        "original_line_index": sample.get("original_line_index"),
        "gold_output": sample.get("output"),
        "raw_prediction": None,
        "decoded_text": None,
        "parsed_prediction": None,
        "generate_success": False,
        "decode_success": False,
        "parse_success": False,
        "is_correct": False,
        "latency_sec": None,
        "tokens_per_sec": None,
        "input_tokens": None,
        "actual_new_tokens": None,
        "gpu_memory_before": None,
        "gpu_memory_after": None,
        "gpu_peak_memory": None,
        "error_stage": "none",
        "error_message": None,
        "failure_reason": None,
        "bottleneck_type": None,
    }


def load_failure_record(sample: dict[str, Any], group: str, message: str | None) -> dict[str, Any]:
    record = base_record(sample, group)
    record["error_stage"] = "model_loading"
    record["error_message"] = message
    record["bottleneck_type"] = classify_bottleneck("model_loading", message, None)
    record["failure_reason"] = concise_failure_reason("model_loading", message, None)
    return record


def run_sample(model: Any, processor: Any, tokenizer: Any, sample: dict[str, Any], group: str) -> dict[str, Any]:
    import torch

    record = base_record(sample, group)
    started = time.perf_counter()
    try:
        inputs, token_info = build_inputs(processor, tokenizer, sample)
        record.update(token_info)
        record["input_tokens"] = token_info["input_tokens_metric"]
        inputs = move_inputs_to_device(inputs, first_model_device(model))
        record["gpu_memory_before"] = gpu_memory_snapshot()
        sync_cuda()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        sync_cuda()
        latency = time.perf_counter() - started
        record["generate_success"] = True
        decoded, decode_ok, decode_error, actual_new_tokens = decode_prediction(outputs, inputs, tokenizer, processor)
        record["actual_new_tokens"] = actual_new_tokens
        record["latency_sec"] = round(latency, 3)
        if actual_new_tokens and latency > 0:
            record["tokens_per_sec"] = round(actual_new_tokens / latency, 3)
        record["gpu_memory_after"] = gpu_memory_snapshot()
        record["gpu_peak_memory"] = gpu_peak_snapshot()
        if not decode_ok:
            record["error_stage"] = "decode"
            record["error_message"] = decode_error
            record["bottleneck_type"] = classify_bottleneck("decode", decode_error, actual_new_tokens)
            record["failure_reason"] = concise_failure_reason("decode", decode_error, actual_new_tokens)
            return record
        record["decode_success"] = True
        record["decoded_text"] = decoded
        record["raw_prediction"] = decoded

        task = sample_task(sample)
        pred, pred_err = parse_answer(decoded, task)
        gold, gold_err = parse_answer(sample.get("output"), task)
        if pred_err or gold_err:
            record["error_stage"] = "parse"
            record["error_message"] = pred_err or gold_err
            record["bottleneck_type"] = classify_bottleneck("parse", record["error_message"], actual_new_tokens)
            record["failure_reason"] = concise_failure_reason("parse", record["error_message"], actual_new_tokens)
            return record
        record["parse_success"] = True
        record["parsed_prediction"] = pred
        record["is_correct"] = is_correct_prediction(pred, gold, task)
        if actual_new_tokens is not None and actual_new_tokens < MAX_NEW_TOKENS:
            record["bottleneck_type"] = "输入/生成瓶颈"
            record["failure_reason"] = concise_failure_reason("none", None, actual_new_tokens)
        return record
    except Exception as exc:
        latency = time.perf_counter() - started
        record["latency_sec"] = round(latency, 3)
        record["gpu_memory_after"] = gpu_memory_snapshot()
        record["gpu_peak_memory"] = gpu_peak_snapshot()
        record["error_stage"] = "generate"
        record["error_message"] = f"{exc.__class__.__name__}: {exc}"
        record["bottleneck_type"] = classify_bottleneck("generate", record["error_message"], record.get("actual_new_tokens"))
        record["failure_reason"] = concise_failure_reason("generate", record["error_message"], record.get("actual_new_tokens"))
        return record


def summarize_records(
    spec: ConfigSpec,
    env: dict[str, Any],
    load_success: bool,
    load_error: str | None,
    load_time: float | None,
    load_info: dict[str, Any],
    records_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_records = [item for records in records_by_group.values() for item in records]
    main = records_by_group.get("main", [])
    paper = records_by_group.get("paper", [])
    stress = records_by_group.get("stress", [])
    generated = [r for r in all_records if r["generate_success"]]
    decoded = [r for r in all_records if r["decode_success"]]
    parsed = [r for r in all_records if r["parse_success"]]
    main_parsed = [r for r in main if r["parse_success"]]
    paper_parsed = [r for r in paper if r["parse_success"]]
    stress_parsed = [r for r in stress if r["parse_success"]]
    latencies = [float(r["latency_sec"]) for r in generated if isinstance(r.get("latency_sec"), (int, float))]
    tps = [float(r["tokens_per_sec"]) for r in generated if isinstance(r.get("tokens_per_sec"), (int, float))]
    input_tokens = [int(r["input_tokens"]) for r in all_records if isinstance(r.get("input_tokens"), int)]
    new_tokens = [int(r["actual_new_tokens"]) for r in generated if isinstance(r.get("actual_new_tokens"), int)]
    bottlenecks = Counter(r.get("bottleneck_type") for r in all_records if r.get("bottleneck_type"))
    peak_memory: dict[str, dict[str, float]] = {}
    final_memory: dict[str, dict[str, float]] = {}
    for record in all_records:
        for gpu, stats in (record.get("gpu_peak_memory") or {}).items():
            peak_memory.setdefault(gpu, {"max_allocated_gib": 0.0, "max_reserved_gib": 0.0})
            for key in ("max_allocated_gib", "max_reserved_gib"):
                peak_memory[gpu][key] = max(peak_memory[gpu][key], float(stats.get(key) or 0.0))
        if record.get("gpu_memory_after"):
            final_memory = record["gpu_memory_after"]

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    return {
        "config": spec.name,
        "label": spec.label,
        "model": MODEL_NAME,
        "batch_size": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "precision": spec.precision,
        "cuda_visible_devices": spec.cuda_visible_devices,
        "requested_device_map": device_map_for(spec),
        "environment": env,
        "load_success": load_success,
        "load_error": load_error,
        "load_time_sec": round(load_time, 3) if load_time is not None else None,
        "load_info": load_info,
        "counts": {group: len(records) for group, records in records_by_group.items()},
        "official_denominator_note": "正式成功率统计三组样例合计：主测试 20 + 论文样例 4 + 压力测试 1。",
        "official_accuracy": rate(sum(1 for r in all_records if r.get("is_correct")), len(all_records)),
        "official_generate_success_rate": rate(len(generated), len(all_records)),
        "official_decode_success_rate": rate(len(decoded), len(all_records)),
        "official_parse_success_rate": rate(len(parsed), len(all_records)),
        "generate_success_rate_all": rate(len(generated), len(all_records)),
        "decode_success_rate_all": rate(len(decoded), len(all_records)),
        "parse_success_rate_all": rate(len(parsed), len(all_records)),
        "main_accuracy": rate(sum(1 for r in main if r.get("is_correct")), len(main)),
        "main_parse_accuracy": rate(sum(1 for r in main_parsed if r.get("is_correct")), len(main_parsed)),
        "main_generate_success_rate": rate(sum(1 for r in main if r["generate_success"]), len(main)),
        "main_decode_success_rate": rate(sum(1 for r in main if r["decode_success"]), len(main)),
        "main_parse_success_rate": rate(sum(1 for r in main if r["parse_success"]), len(main)),
        "paper_accuracy": rate(sum(1 for r in paper if r.get("is_correct")), len(paper)),
        "paper_parse_accuracy": rate(sum(1 for r in paper_parsed if r.get("is_correct")), len(paper_parsed)),
        "stress_accuracy": rate(sum(1 for r in stress if r.get("is_correct")), len(stress)),
        "stress_parse_accuracy": rate(sum(1 for r in stress_parsed if r.get("is_correct")), len(stress_parsed)),
        "avg_input_tokens": round(sum(input_tokens) / len(input_tokens), 3) if input_tokens else None,
        "avg_actual_new_tokens": round(sum(new_tokens) / len(new_tokens), 3) if new_tokens else None,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "max_latency_sec": round(max(latencies), 3) if latencies else None,
        "avg_tokens_per_sec": round(sum(tps) / len(tps), 3) if tps else None,
        "final_memory": final_memory or gpu_memory_snapshot(),
        "peak_memory": peak_memory or gpu_peak_snapshot(),
        "bottleneck_counts": dict(bottlenecks),
        "failure_count_by_stage": dict(Counter(r["error_stage"] for r in all_records if r["error_stage"] != "none")),
        "first_error": next((r["failure_reason"] or r["error_message"] for r in all_records if r.get("failure_reason") or r.get("error_message")), load_error),
    }


def run_config(config_name: str) -> int:
    if config_name not in CONFIGS:
        raise ValueError(f"未知配置: {config_name}")
    validate_inputs()
    spec = CONFIGS[config_name]
    out_dir = RESULT_ROOT / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(out_dir / "run.log")
    records_by_group: dict[str, list[dict[str, Any]]] = {"main": [], "paper": [], "stress": []}
    data_groups = {
        "main": (MAIN_DATA, out_dir / "main_predictions.jsonl"),
        "paper": (PAPER_DATA, out_dir / "paper_predictions.jsonl"),
        "stress": (STRESS_DATA, out_dir / "stress_predictions.jsonl"),
    }
    for _, output_path in data_groups.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    load_success = False
    load_error = None
    load_time = None
    load_info: dict[str, Any] = {}
    try:
        set_hf_cache_env()
        reset_gpu_peak()
        env = gpu_environment()
        logger.log(f"开始运行配置: {spec.name} / {spec.label}")
        logger.log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        model, processor, tokenizer, load_time, load_info = load_model(spec, logger)
        load_info["merge_patch_applied"] = apply_timeseries_merge_patch(model, logger)
        load_success = True
        for group, (path, output_path) in data_groups.items():
            rows = load_jsonl(path)
            for idx, sample in enumerate(rows, start=1):
                logger.log(f"[{group} {idx}/{len(rows)}] {sample.get('sample_id')}")
                record = run_sample(model, processor, tokenizer, sample, group)
                records_by_group[group].append(record)
                append_jsonl(output_path, record)
                logger.log(f"  stage={record['error_stage']} correct={record['is_correct']} latency={record['latency_sec']}")
    except Exception as exc:
        load_error = f"{exc.__class__.__name__}: {exc}"
        logger.exception("配置运行失败", exc)
        for group, (path, output_path) in data_groups.items():
            if records_by_group[group]:
                continue
            rows = load_jsonl(path)
            for sample in rows:
                record = load_failure_record(sample, group, load_error)
                records_by_group[group].append(record)
                append_jsonl(output_path, record)
    finally:
        env = gpu_environment()
        summary = summarize_records(spec, env, load_success, load_error, load_time, load_info, records_by_group)
        write_json(out_dir / "summary.json", summary)
        write_config_report(spec, summary)
        logger.log("summary:")
        logger.log(json.dumps(summary, ensure_ascii=False, indent=2))
        logger.close()
    return 0 if load_success else 1


def run_all() -> int:
    failures = 0
    for config_name, spec in CONFIGS.items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
        cmd = [sys.executable, str(SCRIPT_PATH), "run-config", "--config", config_name]
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            failures += 1
    write_filled_summary_if_possible()
    return 0 if failures == 0 else 1


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_config_report(spec: ConfigSpec, summary: dict[str, Any]) -> None:
    report = f"""# 实验一：{spec.label}

## 固定配置

- 模型：`STReasoner_8B` / `{MODEL_NAME}`
- batch size：1
- max_new_tokens：{MAX_NEW_TOKENS}
- precision：`{spec.precision}`
- CUDA_VISIBLE_DEVICES：`{spec.cuda_visible_devices}`
- requested device_map：`{summary.get('requested_device_map')}`

## 运行摘要

- load 成功：{summary.get('load_success')}
- load 错误：{summary.get('load_error')}
- 正式 generate 成功率（三组合计）：{summary.get('official_generate_success_rate')}
- 正式 decode 成功率（三组合计）：{summary.get('official_decode_success_rate')}
- 正式 parse 成功率（三组合计）：{summary.get('official_parse_success_rate')}
- 正式平均正确率（三组合计）：{summary.get('official_accuracy')}
- 主测试平均正确率：{summary.get('main_accuracy')}
- 论文样例平均正确率：{summary.get('paper_accuracy')}
- 压力测试平均正确率：{summary.get('stress_accuracy')}
- 平均 input tokens：{summary.get('avg_input_tokens')}
- 平均 actual new tokens：{summary.get('avg_actual_new_tokens')}
- 平均延迟：{summary.get('avg_latency_sec')} 秒
- 最高延迟：{summary.get('max_latency_sec')} 秒
- 平均 tokens/s：{summary.get('avg_tokens_per_sec')}
- 峰值显存：`{json.dumps(summary.get('peak_memory'), ensure_ascii=False)}`
- 瓶颈统计：`{json.dumps(summary.get('bottleneck_counts'), ensure_ascii=False)}`

## 三组样例说明

- 主测试样例：计入正式成功率，并单独保留分组指标。
- 论文样例：计入正式成功率，并单独保留分组指标。
- 压力测试样例：计入正式成功率，并单独保留分组指标。

机器可读结果见：`{rel(RESULT_ROOT / spec.name)}`
"""
    (DOC_ROOT / f"experiment1_{spec.name}.md").write_text(report, encoding="utf-8")


def write_filled_summary_if_possible() -> None:
    summaries: dict[str, dict[str, Any]] = {}
    for config_name in CONFIGS:
        path = RESULT_ROOT / config_name / "summary.json"
        if path.exists():
            summaries[config_name] = json.loads(path.read_text(encoding="utf-8"))
    if not summaries:
        return
    validation = validate_inputs()

    def cell(config_name: str, key: str) -> str:
        return fmt(summaries.get(config_name, {}).get(key))

    def load_info_cell(config_name: str, key: str) -> str:
        return fmt(summaries.get(config_name, {}).get("load_info", {}).get(key))

    cols = ["4bit_single", "8bit_single", "fp16_single", "fp16_dual"]
    rows = [
        ("配置证据", "加载方式", [CONFIGS[c].precision for c in cols]),
        ("", "device_map", [fmt(summaries.get(c, {}).get("requested_device_map")) for c in cols]),
        ("", "实际模型分布", [load_info_cell(c, "actual_device_map") for c in cols]),
        ("", "is_cpu_offload", ["见 actual_device_map" for _ in cols]),
        ("", "use_cache", [load_info_cell(c, "use_cache") for c in cols]),
        ("可运行证据", "input tokens（平均值）", [cell(c, "avg_input_tokens") for c in cols]),
        ("", "actual new tokens（平均值）", [cell(c, "avg_actual_new_tokens") for c in cols]),
        ("", "load 成功率", [cell(c, "load_success") for c in cols]),
        ("", "generate 成功率", [cell(c, "official_generate_success_rate") for c in cols]),
        ("资源与速度", "GPU 总显存", [fmt(summaries.get(c, {}).get("environment", {}).get("gpu_total_gib")) for c in cols]),
        ("", "load 后显存", [load_info_cell(c, "load_after_memory") for c in cols]),
        ("", "generate 峰值显存", [cell(c, "peak_memory") for c in cols]),
        ("", "平均延迟与最高延迟", [f"{cell(c, 'avg_latency_sec')} / {cell(c, 'max_latency_sec')}" for c in cols]),
        ("", "tokens/s", [cell(c, "avg_tokens_per_sec") for c in cols]),
        ("", "decode 成功率", [cell(c, "official_decode_success_rate") for c in cols]),
        ("", "parse 成功率", [cell(c, "official_parse_success_rate") for c in cols]),
        ("", "平均正确率（对比失败也算错误）", [cell(c, "official_accuracy") for c in cols]),
        ("失败阶段、失败原因", "失败阶段、详细失败原因；若有输出，输出是否正确（T/F）", [cell(c, "first_error") for c in cols]),
    ]
    table_lines = [
        "|           | dtype                        | 4bit单卡 | 8bit单卡 | fp16单卡 | fp16双卡 |",
        "| --------- | ---------------------------- | -------: | -------: | -------: | ------- |",
    ]
    for section, metric, values in rows:
        table_lines.append("| " + " | ".join([section, metric, *values]) + " |")

    summary_doc = f"""# 实验一：不同精度推理资源测试

## 样本与目录

- 主测试样例：`{rel(MAIN_DATA)}`，共 {validation['main_count']} 条，四类任务各 5 条。
- 论文样例：`{rel(PAPER_DATA)}`，共 {validation['paper_count']} 条。
- 压力测试样例：`{rel(STRESS_DATA)}`，共 {validation['stress_count']} 条。
- 运行脚本：`{rel(SCRIPT_PATH)}`
- 机器可读结果目录：`{rel(RESULT_ROOT)}`

{evidence_section()}

## 实验记录表

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|       样本       | 主测试 20 + 论文样例 4 + 压力测试 1 |
|   batch size   |                1                |
| max_new_tokens |               512               |

{chr(10).join(table_lines)}

## 正式统计口径

正式成功率统计三组样例合计 25 条：主测试 20 条 + 论文样例 4 条 + 压力测试 1 条。报告中仍保留三组样例的分组结果用于定位问题。

## 瓶颈类型总结

详见各配置报告与 `summary.json` 中的 `bottleneck_counts`。失败原因统一归入资源瓶颈、输入/生成瓶颈、输出与评测瓶颈三类；确定的原因尽量细写，不确定时只写粗略阶段和错误信息。
"""
    (DOC_ROOT / "experiment_summary.md").write_text(summary_doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一：不同精度推理资源测试")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prepare", help="只生成证据和空表，不运行模型")
    run_config_parser = sub.add_parser("run-config", help="运行单个配置")
    run_config_parser.add_argument("--config", choices=CONFIGS.keys(), required=True)
    sub.add_parser("run-all", help="依次运行四组配置")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "prepare"
    if command == "prepare":
        write_prepare_docs()
        print(f"已生成空表和证据文档: {DOC_ROOT / 'experiment_summary.md'}")
        return 0
    if command == "run-config":
        return run_config(args.config)
    if command == "run-all":
        return run_all()
    raise ValueError(f"未知命令: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
