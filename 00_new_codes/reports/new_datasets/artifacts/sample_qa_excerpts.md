# HeaRTS 真实样例摘录（HF frozen test cases）

GT and glucose_stats from pickle.load when available; pickletools fallback. Prompt from official task source.

### 样例 1：`cgmacros` / `cgm_stat_calculation` / `0.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 3.7037}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 0

### 样例 2：`cgmacros` / `cgm_stat_calculation` / `1.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 17.051}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 1

### 样例 3：`cgmacros` / `cgm_stat_calculation` / `2.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 8.246}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 2

### 样例 4：`cgmacros` / `cgm_stat_calculation` / `3.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 17.3305}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 3

### 样例 5：`cgmacros` / `cgm_stat_calculation` / `4.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 15.0245}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 4

### 样例 6：`cgmacros` / `cgm_stat_calculation` / `5.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 15.1642}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 5

### 样例 7：`cgmacros` / `cgm_stat_calculation` / `6.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 6.8484}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 6

### 样例 8：`cgmacros` / `cgm_stat_calculation` / `7.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 7.4773}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 7

### 样例 9：`cgmacros` / `cgm_stat_calculation` / `8.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 15.0943}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 8

### 样例 10：`cgmacros` / `cgm_stat_calculation` / `9.pkl`
- **能力维**：Perception · Stat. Calculation
- **题干/任务（原文）**：

The continuous glucose monitors (CGM) data for this subject is provided in 'input/cgm.csv'. There are two columns in this csv file: one is timestamp containing the time of each reading, and the other column "Libre GL" contains glucose values (mg/dL). Calculate percentage of time CGM is below and above normal range (70 - 180 mg/dL). Please calculate and output your final answer as a JSON object without any other text in the following format:
{
    "below": [float, percentage of time CGM < 70 mg/dL],
    "above": [float, percentage of time CGM > 180 mg/dL]
}

- **题干来源**：HEARTS/exp/cgmacros/cgm_stat_calculation.py run_agent prompt (official)
- **输入**：CGM time series as pandas DataFrame in pickle key 'cgm'; columns include timestamp + Libre GL (mg/dL); ~1431 rows per minute sampling.
- **正确答案（GT）**：`{"below": 0.0, "above": 0.0}`
- **评测指标**：sMAPE (1-0.5*sMAPE)
- **备注**：testcase file index 9
