#!/usr/bin/env python3
"""E4: stratified fixed idx lists for external Tier1 experiments (CPU ledger)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from compute import OUT, RUNS, TASKS, load_jsonl, load_mismatch_indices  # noqa: E402
from compute_e2_strict_reparse import load_strict_preds  # noqa: E402

ART = OUT


def main() -> None:
    rfl = json.loads((ART / "real_instability_flip_ledger.json").read_text(encoding="utf-8"))
    ga = json.loads((ART / "graph_ablation_sample_ledger.json").read_text(encoding="utf-8"))
    e2 = json.loads((ART / "e2_strict_reparse_summary.json").read_text(encoding="utf-8"))
    mm1 = load_mismatch_indices("run1")

    parse_fail_any: dict[str, list[int]] = {}
    for task in TASKS:
        if task not in e2["parse_fail"]:
            continue
        n = e2["parse_fail"][task]["n"]
        ok_maps = {rn: load_strict_preds(task, RUNS[rn])[1] for rn in RUNS}
        parse_fail_any[task] = [
            idx for idx in range(n) if any(not ok_maps[rn].get(idx, False) for rn in RUNS)
        ]

    out = {
        "purpose": "External Tier1 shared stratified idx lists (superpowers CPU ledger)",
        "correlation_real_instability_flip": rfl["correlation"]["real_instability_indices"],
        "correlation_parser_induced_flip": rfl["correlation"]["parser_induced_indices"],
        "entity_real_instability_flip": rfl["entity"]["real_instability_indices"],
        "etiological_flip_run123": rfl["etiological"]["real_instability_indices"],
        "parse_fail_any_run": parse_fail_any,
        "graph_rescued": {
            row["task"]: row["rescued_sample_indices"] for row in ga["tasks"]
        },
        "persistent_mismatch_run1": {t: sorted(mm1[t]) for t in mm1},
        "wave6_unclosed_correlation": [191, 384, 46, 78, 399, 463, 619, 628, 640, 652, 669, 698],
    }
    path = ART / "stratified_idx_lists_e4.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
