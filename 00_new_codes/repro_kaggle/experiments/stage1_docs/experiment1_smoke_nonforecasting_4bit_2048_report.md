# Experiment 1 Smoke Report: 4bit Non-Forecasting, max_new_tokens=2048

Generated at: 2026-05-19 18:10:21 UTC

## Scope

- Script: `repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py`
- Result directory: `repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048`
- Config: `4bit_single`
- Model: `Time-HD-Anonymous/STReasoner-8B`
- Sample: `tiny20_entity_01_line257`
- Task: `entity` / official task `reasoning_entity`
- Source file: `ST-Test/entity_test.jsonl`
- Original line index: `257`
- Local index in main tiny20 set: `5`
- Batch size: `1`
- `max_new_tokens`: `2048`

## Question And Gold

Question:

```text
Which (name, description) pair should Node 2 correspond to?
Options:
A. (Flow Junction, Storm drain outlet)
B. (Distribution Hub, Secondary water channel)
C. (Confluence Point, Main river confluence)
D. (Collection Basin, Reservoir intake point)
```

Gold output:

```text
<answer>C</answer>
```

Model semantic answer:

```text
C. (Confluence Point, Main river confluence)
```

## A. Run Layer Metrics

### Model Loading

- Load success: `true`
- Load time: `114.562 sec`
- Processor class: `Qwen3TSProcessor`
- Tokenizer class: `Qwen2TokenizerFast`
- Model class: `Qwen3TSForCausalLM`
- Requested device map: `{"": 0}`
- Actual device map: `{"": 0}`
- Model distribution: single GPU, CUDA device `0`
- CPU offload: `false`
- Disk offload: `false`
- `use_cache`: `false`
- First parameter dtype: `torch.float16`
- Runtime time-series merge patch applied: `true`
- Load-after memory:
  - GPU0 allocated: `5.703 GiB`
  - GPU0 reserved: `6.740 GiB`

### Input And Generation

- Generate success: `true`
- Decode success: `true`
- Input token metric: `1182`
- Tokenizer prompt tokens: `332`
- Processor input ids length: `972`
- Estimated time-series patch tokens: `210`
- Actual generated new tokens: `1211`
- Stopped before max: `true`
- Generation latency: `1816.342 sec`
- Throughput: `0.667 tokens/sec`
- Bottleneck label recorded by script: `输入/生成瓶颈`
- Failure/reason note recorded by script: generation ended early because `actual_new_tokens=1211 < max_new_tokens=2048`; likely EOS or natural stop.

### Per-Sample GPU Memory

- Memory before generate:
  - GPU0 allocated: `5.703 GiB`
  - GPU0 reserved: `6.740 GiB`
- Memory after generate:
  - GPU0 allocated: `5.711 GiB`
  - GPU0 reserved: `6.744 GiB`
- Per-sample peak during generate:
  - GPU0 max allocated: `5.989 GiB`
  - GPU0 max reserved: `6.744 GiB`

### Runtime Observation

- During the long generate call, GPU0 stayed at roughly `97-100%` utilization.
- Memory stayed around `7.0 GiB` used in `nvidia-smi`.
- This indicates the run was actively computing, not hanging in I/O, model loading, file writing, strict diagnostics, or official evaluation.
- Main runtime bottleneck for this smoke test is the single-sample `model.generate` call with `use_cache=false` and a high `max_new_tokens` cap.

## B. Strict Diagnostic Layer

- Strict format success: `false`
- Strict error: `expected_exactly_one_answer_tag_got_0`
- Parsed value: `null`
- Answer tag count: `0`
- Required ideal format for this choice task: `<answer>C</answer>`

Important distinction:

- The model did answer `C` in natural language.
- The model did not output any `<answer>...</answer>` tag.
- Therefore this is an output-format failure under the strict diagnostic layer.

Notable decoded-text artifacts:

- The model repeated its explanation multiple times.
- The model emitted a stray `ś`.
- The model emitted a stray `</think>`.
- The final answer appears as Markdown-style text: `**Answer: C. (Confluence Point, Main river confluence)**`.

## C. Official Eval Layer

Official eval was run through `evaluation/evaluate_qa.py` using generated files under:

- `official_eval/all/reasoning_entity/dataset.jsonl`
- `official_eval/all/reasoning_entity/generated_answer_new.json`
- `official_eval/main/reasoning_entity/dataset.jsonl`
- `official_eval/main/reasoning_entity/generated_answer_new.json`

Official metrics:

- Scope `all`, task `reasoning_entity`:
  - Total samples: `1`
  - Evaluated samples: `1`
  - Missing predictions: `0`
  - Coverage: `1.0`
  - Accuracy: `0.0`
- Scope `main`, task `reasoning_entity`:
  - Total samples: `1`
  - Evaluated samples: `1`
  - Missing predictions: `0`
  - Coverage: `1.0`
  - Accuracy: `0.0`

Interpretation:

- Official evaluation consumed the prediction file successfully.
- Coverage is `1.0`, so the prediction was present.
- Accuracy is `0.0` because the author parser/evaluator did not extract the intended answer from this verbose output.
- This confirms the new script's official-eval layer can run end to end, while the model output format remains a correctness bottleneck.

## Output Excerpt

The decisive answer phrase in the decoded model output was:

```text
**Answer: C. (Confluence Point, Main river confluence)**
```

The final repeated answer phrase was:

```text
**Answer: C. (Confluence Point, Main river confluence)**
```

No `<answer>C</answer>` substring appeared in the decoded output.

## Takeaways

- The new three-layer script path is operational for this non-forecasting smoke case.
- The 4bit model loads on one T4 without CPU/disk offload.
- Per-sample peak generation memory is recorded after resetting CUDA peak stats immediately before `generate`.
- The run is slow: about `30.27 min` for one generated sample and `1211` new tokens.
- The biggest observed runtime factor is generation, plausibly worsened by `use_cache=false`.
- The model's semantic answer is correct, but both strict diagnostic and official evaluation fail because the output is not in the expected machine-readable answer format.

## 4bit Rerun Stopped At End Of Day

记录时间：2026-05-19 18:47:09 UTC

- Rerun directory: `repro_kaggle/experiments/stage1_results/experiment1_precision_resource/4bit_single/smoke_nonforecasting_4bit_2048_rerun_20260519_182957`
- Config: `4bit_single`
- Sample: `tiny20_entity_01_line257`
- `max_new_tokens`: `2048`
- CUDA_VISIBLE_DEVICES: `0`
- PYTORCH_CUDA_ALLOC_CONF: `expandable_segments:True`
- Load success: true
- Load time: `57.190 sec`
- Load-after memory:
  - GPU0 allocated: `5.690 GiB`
  - GPU0 reserved: `6.748 GiB`
- Stopped stage: `STAGE=generate_one_sample`
- Stop reason: user ended the day's runs; process received `KeyboardInterrupt`.
- Final partial state: `stage = "failed"`, `error_type = "KeyboardInterrupt"`.
- No final answer, strict diagnostic, or official eval was produced for this rerun.

## 8bit Same-Sample Attempt

An 8bit comparison was started with the same sample and generation cap:

- Result directory: `repro_kaggle/experiments/stage1_results/experiment1_precision_resource/8bit_single/smoke_nonforecasting_8bit_2048`
- Config: `8bit_single`
- Sample: `tiny20_entity_01_line257`
- Task: `entity` / official task `reasoning_entity`
- `max_new_tokens`: `2048`
- CUDA visible devices: `0`
- GPU: one Tesla T4, `14.563 GiB` visible total memory

### 8bit Model Loading

- Load success: `true`
- Load time: `67.987 sec`
- Processor class: `Qwen3TSProcessor`
- Tokenizer class: `Qwen2TokenizerFast`
- Model class: `Qwen3TSForCausalLM`
- Quantization: `BitsAndBytesConfig(load_in_8bit=True)`
- Requested device map: `{"": 0}`
- Actual device map: `{"": 0}`
- Model distribution: single GPU, CUDA device `0`
- CPU offload: `false`
- Disk offload: `false`
- `use_cache`: `false`
- First parameter dtype: `torch.float16`
- Runtime time-series merge patch applied: `true`
- Load-after memory:
  - GPU0 allocated: `8.859 GiB`
  - GPU0 reserved: `9.064 GiB`

### 8bit Generate Status

- Entered stage: `STAGE=generate_one_sample`
- No normal `record.run` output was produced.
- No `SMOKE_DONE` marker was written.
- No Python exception was captured by the script logger.
- `partial_result.json` remained at stage `model_loaded`.
- `result.json` was not created.
- The PTY session exited with code `1`.
- After exit, `nvidia-smi` showed no running GPU process and `0 MiB` used on both GPUs.

Observed while generate was running:

- After about 5 minutes of generate:
  - GPU0 memory used in `nvidia-smi`: about `11521 MiB`
  - GPU utilization: about `91%`
- Shortly before discovering the exit:
  - GPU0 memory used in `nvidia-smi`: about `11885 MiB`
  - GPU utilization: about `68%`
  - Python process showed as defunct, indicating it had already exited but had not been reaped at that instant.

Interpretation:

- 8bit did not fail at model loading.
- 8bit used substantially more memory than 4bit:
  - 4bit load-after allocated: `5.703 GiB`
  - 8bit load-after allocated: `8.859 GiB`
  - Difference: `+3.156 GiB`
- The 8bit failure occurred during or inside the generate phase.
- Because the script did not catch an exception and did not write a generate failure record, this looks like an external/native-process termination or lower-level failure rather than a normal Python exception from `run_one_sample`.
- The immediate reproducible comparison point is therefore: 4bit completes the same sample, while 8bit loads successfully but does not complete generation under the same one-T4, `max_new_tokens=2048` setting.
