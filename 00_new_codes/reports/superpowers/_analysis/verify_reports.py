"""Verify key report numbers match artifacts (verification-before-completion)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(name: str):
    with open(ART / name, encoding="utf-8") as f:
        return json.load(f)


def check(label: str, got, expected):
    ok = got == expected
    print(f"{'OK' if ok else 'FAIL'} {label}: got={got!r} expected={expected!r}")
    return ok


def main() -> int:
    ok = True
    tb = load("token_budget_512_vs_6144.json")
    ok &= check("corr flip", tb["correlation"]["flip_count"], 163)
    ok &= check("entity flip", tb["entity"]["flip_count"], 223)
    ok &= check("eti flip 512v6144", tb["etiological"]["flip_count"], 14)

    ga = load("graph_ablation_sample_ledger.json")
    ok &= check("graph corr rescued", ga["summary"]["correlation"], 161)
    ok &= check("graph entity rescued", ga["summary"]["entity"], 226)
    ok &= check("graph eti rescued", ga["summary"]["etiological"], 9)

    eti = load("etiological_and_paper_cases.json")
    ok &= check("eti 3run flip", eti["etiological"]["flip_count"], 27)
    ok &= check("paper cases n", eti["paper_cases"]["paper_cases_count"], 4)

    kg = load("kaggle_precision_audit.json")
    ok &= check("kaggle configs", len(kg["configs"]), 4)
    ok &= check("kaggle tags sum", sum(c["with_answer_tag"] for c in kg["configs"]), 0)

    reg = load("experiment_registry.json")
    n = len(reg) if isinstance(reg, list) else len(reg.get("entries", reg))
    ok &= check("experiment_registry entries", n, 71)

    cw = load("code_experiment_crosswalk.json")
    ok &= check("crosswalk rows", len(cw["rows"]), 5)

    e2 = load("e2_strict_reparse_summary.json")
    ok &= check("e2 corr loose triple", e2["falsifier_check"]["correlation_e2_loose"], 67)
    ok &= check("e2 corr strict triple", e2["falsifier_check"]["correlation_e2_strict"], 18)
    ok &= check("e2 corr strict flip", e2["flip_comparison"]["correlation"]["strict_flip"], 173)
    ok &= check("e2 corr parse_fail run1", e2["parse_fail"]["correlation"]["per_run"]["run1"]["parse_fail"], 47)

    t0 = load("t0_2_parse_dashboard.json")
    ok &= check("t0 corr parser flip", t0["flip_attribution"]["correlation"]["parser_induced_flip_count"], 88)
    ok &= check("t0 corr strict flip", t0["flip_attribution"]["correlation"]["strict_flip_count"], 173)
    ok &= check("t0 unclosed subset corr", t0["unclosed_vs_parse_fail"]["correlation"]["focus_subset_of_parse_fail"], True)

    rfl = load("real_instability_flip_ledger.json")
    ok &= check("real flip corr n", len(rfl["correlation"]["real_instability_indices"]), 173)
    ok &= check("real flip corr mismatch0", rfl["correlation"]["real_flip_mismatch_zero"], 132)

    t3 = load("t0_3_num_tokens_audit.json")
    ok &= check("t0-3 ratio median", t3["evidence"]["global_ratio_resp_over_num_tokens_median"], 5.798)

    eti_st = load("etiological_stability.json")
    ok &= check("eti flip count", eti_st["flip_count"], 27)
    ok &= check("eti think median", eti_st["think_len_median"], 2700)

    pc = load("etiological_and_paper_cases.json")["paper_cases"]["cases"][0]
    ok &= check("paper case baseline filled", pc["baseline_6144_run1"], "D")

    e4 = load("stratified_idx_lists_e4.json")
    ok &= check("e4 corr real flip n", len(e4["correlation_real_instability_flip"]), 173)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
