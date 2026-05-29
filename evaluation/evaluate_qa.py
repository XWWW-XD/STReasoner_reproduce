import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


# 把文本转成字符串并去掉前后空格
def _normalize_text(text: Any) -> str:
    return str(text).strip()


# 选择题：优先抽显式最终答案；没有时再退回旧的开头匹配。
def _normalize_choice(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value:
        return ""
    value = _extract_tag_content(value, "answer")

    boxed_matches = re.findall(r"\\boxed\{\s*([A-Da-d])\s*\}", value)
    if boxed_matches:
        return boxed_matches[-1].upper()

    answer_matches = re.findall(
        r"(?:final\s+)?answer\s*[:：]\s*(?:option\s*)?([A-Da-d])\b",
        value,
        flags=re.IGNORECASE,
    )
    if answer_matches:
        return answer_matches[-1].upper()

    option_matches = re.findall(
        r"\boption\s+([A-Da-d])\b[^\n.]{0,160}\b(?:best|correct|accurate|fits|captures|answer|selected|most)\b",
        value,
        flags=re.IGNORECASE,
    )
    if option_matches:
        return option_matches[-1].upper()

    match = re.match(r"\s*([A-Da-d])[\.\)\s-]*", value)
    if match:
        return match.group(1).upper()
    return value.lower()


# 抽取<tag>中内容，没有则保留整段文本
def _extract_tag_content(text: str, tag: str = "answer") -> str:
    if not text:
        return ""
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        answer = answer.replace('```', '').strip()
        return answer
    return text.strip()

# 把 gold 或模型输出解析成 List[float]
def _parse_series(text: Any) -> List[float]:
    if text is None:
        return []
    if isinstance(text, list):
        return [float(v) for v in text]
    if isinstance(text, (int, float)):
        return [float(text)]
    text = str(text).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
        if isinstance(parsed, (int, float)):
            return [float(parsed)]
    except json.JSONDecodeError:
        pass

    prediction_lists = re.findall(
        r'"(?:predictions?|forecast|answer)"\s*:\s*(\[[^\[\]]+\])',
        text,
        flags=re.IGNORECASE,
    )
    for candidate in reversed(prediction_lists):
        parsed = _parse_series(candidate)
        if parsed:
            return parsed

    bracket_lists = re.findall(r"\[[^\[\]]+\]", text)
    for candidate in reversed(bracket_lists):
        numbers = re.findall(r"-?\d+\.?\d*", candidate)
        if numbers and len(numbers) <= 20:
            return [float(n) for n in numbers]

    numbers = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in numbers]


# 读 gold 测试集
def load_jsonl_dataset(path: str) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("idx", idx)
            dataset.append(obj)
    return dataset


# 读模型prediction文件
def load_prediction_files(exp_dir: str, pattern: str = "generated_answer") -> Dict[int, str]:
    predictions: Dict[int, str] = {}
    response_lengths: List[int] = []
    if not os.path.isdir(exp_dir):
        return predictions
    for name in os.listdir(exp_dir):
        if not name.endswith(".json") or pattern not in name:
            continue
        file_path = os.path.join(exp_dir, name)
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (json.JSONDecodeError, OSError):
                continue
        
        # Support new format with metadata wrapper
        if isinstance(entries, dict) and "results" in entries:
            entries = entries["results"]
        
        for entry in entries:
            idx = entry.get("idx")
            if idx is None:
                continue
            texts_for_entry: List[str] = []

            # Format 1: flat structure with a single top-level "response" field
            if "response" in entry:
                texts_for_entry.append(str(entry.get("response", "")))

            # Format 2: COT-style structure with a "responses" list
            responses_field = entry.get("responses")
            if isinstance(responses_field, list) and responses_field:
                # Prefer the first attempt in sorted order by "attempt" (if present)
                sorted_resps = sorted(responses_field, key=lambda r: r.get("attempt", 0))
                chosen = sorted_resps[0]
                if "response" in chosen:
                    texts_for_entry.append(str(chosen.get("response", "")))

            if not texts_for_entry:
                continue

            # Use the first collected text for this entry
            text = texts_for_entry[0]
            response_lengths.append(len(text))
            predictions[idx] = _extract_tag_content(text)

    # Print response length distribution for debugging/analysis
    # if response_lengths:
    #     counter = Counter(response_lengths)
    #     print("\n--- Raw Response Length Distribution (by characters) ---")
    #     print(f"Total responses: {len(response_lengths)}")
    #     print(f"Min length: {min(response_lengths)}, Max length: {max(response_lengths)}")
    #     print("Length -> Count (sorted by length):")
    #     for length in sorted(counter.keys()):
    #         print(f"{length:5d} -> {counter[length]:5d}")
    #     print("--------------------------------------------------------\n")

    return predictions


def evaluate_alignment_predictions(
    dataset: List[Dict[str, Any]], predictions: Dict[int, str]
) -> Dict[str, Any]:
    total = len(dataset)
    evaluated = 0
    missing = 0

    em_total = 0
    em_correct = 0

    rel_total = 0
    rel_sum = 0.0

    overall_sum = 0.0

    for idx, sample in enumerate(dataset):
        target = sample.get("output")
        prediction = predictions.get(idx)
        if prediction is None:
            missing += 1
            continue

        target_float = _safe_float(target)
        pred_float = _safe_float(prediction)
        evaluated += 1

        if target_float is not None and pred_float is not None:
            if abs(target_float) > 1e-6:
                rel_error = abs(pred_float - target_float) / abs(target_float)
            else:
                rel_error = abs(pred_float - target_float)
                rel_score = max(0.0, 1.0 - rel_error)
                rel_sum += rel_score
                rel_total += 1
                overall_sum += rel_score
        else:
            em_total += 1
            if _normalize_text(prediction) == _normalize_text(target):
                em_correct += 1
                overall_sum += 1.0
            else:
                overall_sum += 0.0

    result = {
        "task": "alignment",
        "total_samples": total,
        "evaluated_samples": evaluated,
        "missing_predictions": missing,
        "coverage": evaluated / total if total else 0.0,
        "overall_score": overall_sum / evaluated if evaluated else None,
        "exact_match": em_correct / em_total if em_total else None,
        "relative_accuracy": rel_sum / rel_total if rel_total else None,
    }
    return result


# forecasting任务的测评
def evaluate_forecasting_predictions(
    dataset: List[Dict[str, Any]], predictions: Dict[int, str]
) -> Dict[str, Any]:
    total = len(dataset)
    evaluated = 0
    missing = 0
    missing_idx: List[int] = []
    mae_sum = 0.0
    mape_sum = 0.0
    mape_valid_count = 0
    
    # For calculating average magnitude of target values
    all_target_values: List[float] = []
    all_target_abs_values: List[float] = []

    for idx, sample in enumerate(dataset):
        target_series = _parse_series(sample.get("output"))
        pred_text = predictions.get(idx)
        pred_series = _parse_series(pred_text) if pred_text is not None else []

        if not target_series or not pred_series:
            missing += 1
            missing_idx.append(idx)
            continue

        if len(pred_series) < len(target_series):
            pad_value = pred_series[-1]
            pred_series = pred_series + [pad_value] * (len(target_series) - len(pred_series))
        elif len(pred_series) > len(target_series):
            pred_series = pred_series[: len(target_series)]

        # Collect target values for statistics
        all_target_values.extend(target_series)
        all_target_abs_values.extend([abs(v) for v in target_series])

        absolute_errors = [abs(pred_series[i] - target_series[i]) for i in range(len(target_series))]

        # MAE calculation
        mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
        mae_sum += mae
        
        # MAPE calculation (skip values where target is zero or near-zero to avoid division by zero)
        percentage_errors = []
        for i in range(len(target_series)):
            if abs(target_series[i]) > 1e-8:  # Avoid division by zero
                percentage_errors.append(abs(pred_series[i] - target_series[i]) / abs(target_series[i]))
        
        if percentage_errors:
            mape = sum(percentage_errors) / len(percentage_errors) * 100  # Convert to percentage
            mape_sum += mape
            mape_valid_count += 1
        
        evaluated += 1

    # Calculate target value statistics
    target_mean = sum(all_target_values) / len(all_target_values) if all_target_values else None
    target_abs_mean = sum(all_target_abs_values) / len(all_target_abs_values) if all_target_abs_values else None
    target_min = min(all_target_values) if all_target_values else None
    target_max = max(all_target_values) if all_target_values else None

    result = {
        "task": "reasoning_forecasting",
        "total_samples": total,
        "evaluated_samples": evaluated,
        "missing_predictions": total - evaluated,
        "coverage": evaluated / total if total else 0.0,
        "mae": mae_sum / evaluated if evaluated else None,
        "mape": mape_sum / mape_valid_count if mape_valid_count else None,
        "target_stats": {
            "mean": target_mean,
            "abs_mean": target_abs_mean,
            "min": target_min,
            "max": target_max,
            "total_values": len(all_target_values),
        },
        "missing_indices": missing_idx,
    }
    return result


# 多选题的测评
def evaluate_multiple_choice_predictions(
    dataset: List[Dict[str, Any]], predictions: Dict[int, str], task: str
) -> Dict[str, Any]:
    total = len(dataset)
    evaluated = 0
    correct = 0
    missing_indices: List[int] = []

    for idx, sample in enumerate(dataset):
        target = sample.get("output")
        prediction = predictions.get(idx)
        if prediction is None:
            missing_indices.append(idx)
            continue
        
        evaluated += 1
        # print("prediction:", prediction)
        # print("target:", target)
        # print("normalize_choice(prediction):", _normalize_choice(prediction))
        # print("normalize_choice(target):", _normalize_choice(target))
        # print("--------------------------------")
        if _normalize_choice(prediction) == _normalize_choice(target):
            correct += 1

    # Print some missing indices for debugging
    if missing_indices:
        print(f"\n--- Missing Prediction Indices (showing first 20) ---")
        print(f"Total missing: {len(missing_indices)}")
        print(f"Sample indices: {missing_indices[:20]}")
        print("------------------------------------------------------\n")

    result = {
        "task": task,
        "total_samples": total,
        "evaluated_samples": evaluated,
        "missing_predictions": total - evaluated,
        "coverage": evaluated / total if total else 0.0,
        "accuracy": correct / evaluated if evaluated else None,
    }
    return result


# 任务分发
def evaluate_predictions_for_task(
    dataset: List[Dict[str, Any]], predictions: Dict[int, str], task_type: str
) -> Dict[str, Any]:
    task_type = task_type.lower()
    if task_type == "alignment":
        return evaluate_alignment_predictions(dataset, predictions)
    if task_type == "reasoning_forecasting":
        return evaluate_forecasting_predictions(dataset, predictions)
    if task_type in {"reasoning_entity", "reasoning_etiological", "reasoning_correlation", "reasoning_causal"}:
        return evaluate_multiple_choice_predictions(dataset, predictions, task_type)
    raise ValueError(f"Unsupported task type: {task_type}")
