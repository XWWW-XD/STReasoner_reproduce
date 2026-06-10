#!/usr/bin/env python3
"""Wave-6: unclosed-thinking forensics, strict×mention cross, global parser rates."""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "00_new_codes/tools/mlp_encoder_focused_analysis"))

from compute import (  # noqa: E402
    OUT,
    RUNS,
    THINK_RE,
    collect_sample_outcomes,
    exp_dir,
    load_jsonl,
    response_features,
)
from compute_wave4 import (
    CHOICE_TASKS,
    triple_diverse_indices,
    triple_diverse_indices_loose,
)
from compute_wave5 import classify_run_parser
from count_ts_mentions import count_broad_ts_mentions

ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.S | re.I)
REPEAT_TAIL_RE = re.compile(r"([\d\.]+(?:,\s*[\d\.]+){8,})\s*$")


def load_generated_records(task: str, suffix: str) -> list[dict]:
    path = exp_dir(task, suffix) / "generated_answer.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tail_repetition_score(text: str, window: int = 80) -> dict:
    tail = (text or "")[-window:]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
    if len(nums) < 5:
        return {"unique_ratio": 1.0, "dominant_value": None, "tail_num_count": len(nums)}
    ctr = Counter(nums)
    top_val, top_n = ctr.most_common(1)[0]
    return {
        "unique_ratio": round(len(ctr) / len(nums), 3),
        "dominant_value": top_val,
        "dominant_share": round(top_n / len(nums), 3),
        "tail_num_count": len(nums),
    }


OPEN_THINK_RE = re.compile(r"<think>", re.I)
CLOSE_THINK_RE = re.compile(r"</think>", re.I)


def thinking_body(response: str) -> tuple[str, bool, bool]:
    text = response or ""
    has_open = bool(OPEN_THINK_RE.search(text))
    has_close = bool(CLOSE_THINK_RE.search(text))
    if has_open and has_close:
        m = THINK_RE.search(text)
        return (m.group(1) if m else "", has_open, has_close)
    if has_open:
        start = OPEN_THINK_RE.search(text).end()
        return text[start:], has_open, has_close
    return "", has_open, has_close


def diagnose_response(response: str) -> dict:
    feats = response_features(response)
    think_body, has_open_think, has_close_think = thinking_body(response)
    rep = tail_repetition_score(response)
    root = "unknown"
    if feats["answer_tag_count"] == 0 and has_open_think and not has_close_think:
        if rep["tail_num_count"] >= 8 and rep.get("dominant_share", 0) >= 0.5:
            root = "unclosed_thinking_numeric_loop"
        else:
            root = "unclosed_thinking_other"
    elif feats["answer_tag_count"] == 0:
        root = "missing_answer_no_think_block"
    elif feats["answer_tag_count"] > 1:
        root = "multiple_answer_tags"
    else:
        root = "ok_or_other"
    return {
        "response_len": len(response or ""),
        "answer_tag_count": feats["answer_tag_count"],
        "think_len": len(think_body),
        "has_close_think": has_close_think,
        "has_open_think": has_open_think,
        "root_cause": root,
        **rep,
    }


def all_runs_parser_issue_indices() -> dict[str, list[int]]:
    tax = json.loads((OUT / "parser_anomaly_taxonomy.json").read_text(encoding="utf-8"))
    out = {}
    for task in ["correlation", "entity"]:
        out[task] = [
            s["idx"] for s in tax[task]["samples"] if s["bucket"] == "all_runs_parser_issue"
        ]
    return out


def build_unclosed_thinking_forensics() -> dict:
    """Deep dive on 12+1 all_runs_parser_issue + global no-tag rates."""
    focus = all_runs_parser_issue_indices()
    forensics = {}
    for task in ["correlation", "entity"]:
        rows = []
        for idx in focus[task]:
            per_run = {}
            for rn, suf in RUNS.items():
                recs = {r["idx"]: r for r in load_generated_records(task, suf)}
                resp = recs[idx]["response"]
                per_run[rn] = {
                    **diagnose_response(resp),
                    "input_tokens_field": recs[idx].get("num_tokens"),
                    "parser_class": classify_run_parser(resp, None),
                }
            rows.append({"idx": idx, "runs": per_run})
        forensics[task] = {"count": len(rows), "samples": rows}

    global_rates = {}
    corpus_len = {}
    for task in CHOICE_TASKS:
        global_rates[task] = {}
        corpus_len[task] = {}
        for rn, suf in RUNS.items():
            recs = load_generated_records(task, suf)
            no_tag = 0
            unclosed = 0
            lengths_ok = []
            lengths_bad = []
            for rec in recs:
                resp = rec.get("response", "")
                d = diagnose_response(resp)
                if d["answer_tag_count"] == 0:
                    no_tag += 1
                    lengths_bad.append(d["response_len"])
                else:
                    lengths_ok.append(d["response_len"])
                if d["root_cause"].startswith("unclosed_thinking"):
                    unclosed += 1
            global_rates[task][rn] = {
                "total": len(recs),
                "no_answer_tag": no_tag,
                "no_answer_rate": round(no_tag / len(recs), 4),
                "unclosed_thinking": unclosed,
            }
            corpus_len[task][rn] = {
                "median_len_with_answer": round(statistics.median(lengths_ok), 1) if lengths_ok else None,
                "median_len_no_answer": round(statistics.median(lengths_bad), 1) if lengths_bad else None,
                "mean_len_no_answer": round(statistics.mean(lengths_bad), 1) if lengths_bad else None,
            }

    root_counter = Counter()
    for task in forensics:
        for s in forensics[task]["samples"]:
            for d in s["runs"].values():
                root_counter[d["root_cause"]] += 1

    return {
        "focus_samples": forensics,
        "global_no_answer_rates": global_rates,
        "length_contrast": corpus_len,
        "focus_root_cause_totals": dict(root_counter),
        "interpretation_note": (
            "generated_answer.json num_tokens field stores INPUT token count per inference_tsmllm_vllm.py, "
            "not output length. no_answer_tag samples align 1:1 with unclosed_thinking in this exp."
        ),
    }


def build_strict_triple_mention_cross(outcomes: dict) -> dict:
    datasets = {t: load_jsonl(ROOT / "data/ST-Bench/ST-Test" / f"{t}_test.jsonl") for t in CHOICE_TASKS}
    from compute_wave3 import load_all_responses

    resp = load_all_responses()
    strict = {t: set(triple_diverse_indices(t)) for t in ["correlation", "entity"]}
    loose_noise = {t: set(triple_diverse_indices_loose(t)) - strict[t] for t in ["correlation", "entity"]}

    flip_sets = {t: set() for t in CHOICE_TASKS}
    stable_correct = {t: set() for t in CHOICE_TASKS}
    for key, o in outcomes.items():
        if o["task"] not in CHOICE_TASKS:
            continue
        cor = [o[r]["correct"] for r in RUNS]
        if len(cor) == 3 and len(set(cor)) > 1:
            flip_sets[o["task"]].add(o["idx"])
        if len(cor) == 3 and all(c is True for c in cor):
            stable_correct[o["task"]].add(o["idx"])

    flip_plain = {
        t: flip_sets[t] - strict[t] - loose_noise[t] for t in ["correlation", "entity"]
    }

    def profile(idxs: set[int], task: str) -> dict:
        if not idxs:
            return {"n": 0}
        think_lens = []
        mentions = []
        for idx in idxs:
            full = resp[task]["run1"].get(idx, "")
            think_body, _, _ = thinking_body(full)
            think_lens.append(len(think_body))
            mentions.append(count_broad_ts_mentions(think_body))
        return {
            "n": len(idxs),
            "think_len_median": round(statistics.median(think_lens), 1),
            "think_len_mean": round(statistics.mean(think_lens), 1),
            "broad_mention_median": round(statistics.median(mentions), 1),
            "broad_mention_mean": round(statistics.mean(mentions), 2),
            "zero_mention_share": round(sum(1 for m in mentions if m == 0) / len(mentions), 3),
        }

    result = {}
    for task in ["correlation", "entity"]:
        n = len(datasets[task])
        result[task] = {
            "corpus_run1": profile(set(range(n)), task),
            "strict_triple_diverse": profile(strict[task], task),
            "loose_triple_parser_noise": profile(loose_noise[task], task),
            "flip_not_triple": profile(flip_plain[task], task),
            "stable_all_correct": profile(stable_correct[task], task),
        }
    return result


def build_no_answer_overlap_with_loose_triple() -> dict:
    """How many loose triple-diverse rows are purely parser noise vs real letters."""
    tax = json.loads((OUT / "parser_anomaly_taxonomy.json").read_text(encoding="utf-8"))
    out = {}
    for task in ["correlation", "entity"]:
        buckets = Counter(s["bucket"] for s in tax[task]["samples"])
        out[task] = {
            "loose_not_strict_total": tax[task]["count"],
            "bucket_counts": dict(buckets),
            "all_runs_indices": all_runs_parser_issue_indices()[task],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading...", flush=True)
    outcomes, _ = collect_sample_outcomes()
    payloads = {
        "unclosed_thinking_forensics.json": build_unclosed_thinking_forensics(),
        "strict_triple_mention_cross.json": build_strict_triple_mention_cross(outcomes),
        "loose_triple_noise_breakdown.json": build_no_answer_overlap_with_loose_triple(),
    }
    for name, data in payloads.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
