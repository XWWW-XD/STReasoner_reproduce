"""Shared response diagnostics for superpowers analysis."""
from __future__ import annotations

import re
from collections import Counter

from compute import THINK_RE, response_features

OPEN_THINK_RE = re.compile(r"<think>", re.I)
CLOSE_THINK_RE = re.compile(r"</think>", re.I)


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
