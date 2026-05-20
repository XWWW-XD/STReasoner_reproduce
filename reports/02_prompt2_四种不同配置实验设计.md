请按照下面的实验设计，执行 STReasoner 阶段一的实验一：不同精度推理资源小规模测试。  
  
注意：  
1. 不要先重构实验设计，先按这份设计执行。  
2. 如果发现设计中有明显冲突、代码不支持或字段无法获得，请先报告问题，再给出最小修改建议。  
3. 数据产物和文档产物分开保存：  
- 数据、结果、日志等机器可读产物放在 repro_kaggle/experiments/stage1_subsets 或 stage1_results 下  
- 报告、说明、人工检查记录放在 repro_kaggle/experiments/stage1_docs 下

参考脚本为 /kaggle/working/STReasoner_reproduce/repro_kaggle/experiments/scripts/stage1_script/run_experiment1_new_version.py，可以对脚本内容进行修改，但优先复用。不要新增一套 parser。优先复用参考脚本 run_experiment1_new_version.py 中已有 parser；如果发现它与 evaluation/ 文件夹中的官方评测逻辑不一致，只在报告中说明差异，不要擅自大改 parser。

实验具体设计：
1. 探索以下问题：
	1. 【正式 fp16 推理最低需要什么显存级别的 GPU】精度fp16时，8B模型推理需要多大GPU显存，分单卡和双卡两种情况，速度和输出质量如何？
	2. 【免费 T4 环境下是否可以用 8bit/4bit 作为可行替代方案】精度8bit/4bit时，8B模型推理能不能在T4单卡稳定运行，速度和输出质量如何
	3. 实验设计4组对照，按照从前到后的顺序依次测试：单张4bit，单张8bit，单张fp16，两张fp16

补充：请在最终报告中按“瓶颈类型”总结失败或不稳定原因，不要只写显存。至少区分：
1. 资源瓶颈：显存不足、双卡切分、CPU/disk offload、速度过慢；
2. 输入/生成瓶颈：input tokens 过长、actual new tokens 过短、EOS 提前结束、max_new_tokens 设置影响；
3. 输出与评测瓶颈：decode 异常、输出格式不符合要求、parse 规则不鲁棒、gold answer 对比失败。
4. 按 4bit_single → 8bit_single → fp16_single → fp16_dual 顺序执行。每个配置跑完后先检查该配置的 load/generate/decode、输出文件、关键字段是否正常。如果某个配置出现加载失败、脚本异常、结果文件缺字段、decoded_text 未保存、显存字段缺失等问题，立即停止并报告，不要继续跑后续配置。
5. 实验开始后样例生成比较慢，你可以5分钟看一次或者间隔时间你灵活决定。

报告要求补充：
（1）实验名称统一叫“实验一”，不要再额外使用 tiny20/main20 等作为实验名。不跑 tiny20 全量，不跑 paper cases，不跑 stress。测试样例只使用 SmartTest.jsonl，两条样例：1 条 forecasting，1 条非 forecasting。
SmartTest 来源于：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/st_test_tiny20_seed20260519/tiny20_all.jsonl
SmartTest 输出路径：
repro_kaggle/experiments/stage1_subsets/exp1_resource_tiny20/smart_test/SmartTest.jsonl

（2）总报告只按“四组配置”组织总表：
4bit单卡、8bit单卡、fp16单卡、fp16双卡。

（3）实验报告记录表参考experiment1_summary.md

（3）每个配置下面再写详细报告，并在详细报告最后记录样例的输入、实际输出和正确结果

最终报告路径：
repro_kaggle/experiments/stage1_docs/experiment_summary_2.md
repro_kaggle/experiments/stage1_docs/experiment1_4bit_single.md  
repro_kaggle/experiments/stage1_docs/experiment1_8bit_single.md  
repro_kaggle/experiments/stage1_docs/experiment1_fp16_single.md  
repro_kaggle/experiments/stage1_docs/experiment1_fp16_dual.md

## 实验记录表

|      配置项       |              配置详情               |
| :------------: | :-----------------------------: |
|       模型       |          STReasoner_8B          |
|   batch size   |                1                |
| max_new_tokens |               2048              |

|           | dtype                        |                    4bit单卡 |                    8bit单卡 |                    fp16单卡 | fp16双卡 |
| --------- | ---------------------------- | ------------------------: | ------------------------: | ------------------------: | ------ |
| 配置证据      | 加载方式                         |         单卡 / 双卡 / offload |         单卡 / 双卡 / offload |         单卡 / 双卡 / offload |        |
|           | device_map                   |      单卡 / auto / balanced |      单卡 / auto / balanced |      单卡 / auto / balanced |        |
|           | 实际模型分布                       | 全在 cuda:0 / 分布到两卡 / 有 CPU | 全在 cuda:0 / 分布到两卡 / 有 CPU | 全在 cuda:0 / 分布到两卡 / 有 CPU |        |
|           | is_cpu_offload               |            无 / CPU / disk |            无 / CPU / disk |            无 / CPU / disk |        |
|           | use_cache                    |                默认通常是 True |                默认通常是 True |                默认通常是 True |        |
| 可运行证据     | input tokens（平均值）            |                       实际值 |                       实际值 |                       实际值 |        |
|           | actual new tokens（平均值）       |                       实际值 |                       实际值 |                       实际值 |        |
|           | load 成功率                     |                     成功/失败 |                     成功/失败 |                     成功/失败 |        |
|           | generate 成功率                 |                     成功/失败 |                     成功/失败 |                     成功/失败 |        |
| 资源与速度     | GPU 总显存（若是双卡则分别记录）           |                           |                           |                           |        |
|           | load 后显存（若是双卡则分别记录）          |                      X GB |                      X GB |                      X GB |        |
|           | generate 峰值显存（若是双卡则分别记录）     |                      X GB |                      X GB |                      X GB |        |
|           | 平均延迟与最高延迟                    |                       X 秒 |                       X 秒 |                       X 秒 |        |
|           | tokens/s                     |                         X |                         X |                         X |        |
|           | decode 成功率                   |                     正常/异常 |                     正常/异常 |                     正常/异常 |        |
|           | parse 成功率                    |                     成功/失败 |                     成功/失败 |                     成功/失败 |        |
|           | 平均正确率（对比失败也算错误）              |                           |                           |                           |        |
| 失败阶段、失败原因 | 失败阶段、详细失败原因；若有输出，输出是否正确（T/F） |                           |                           |                           |        |
