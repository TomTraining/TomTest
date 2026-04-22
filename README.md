# TomTest

基于结构化输出的 Theory-of-Mind（心智理论）评测框架，支持多数据集、多模型的基准评测与深度分析。

## 设计理念

**结构化输出优先** - 使用 Pydantic 定义输出 Schema，直接从结构化对象获取答案，避免复杂的字符串解析：

- **新增模型**：只需修改 `experiment_config.yaml` 中的 API 配置，无需改代码
- **新增数据集**：复用现有 Schema，只需编写 `prompts.py` 和 `metrics.py`
- **自动降级**：模型不支持原生结构化输出时，自动切换为 JSON Prompt 注入 + 正则提取

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 主要依赖：openai pyyaml tqdm datasets pyarrow pydantic
```

### 2. 配置实验参数

编辑 `experiment_config.yaml`，填写模型 API 信息：

```yaml
llm:
  model_name: Qwen3-8B           # 模型名称（也是 results/ 下的目录名）
  api_key: not-needed            # 本地服务填 not-needed，云端 API 填密钥
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.6
  max_tokens: 32768
  max_workers: 64
  enable_thinking: false         # Qwen3 等支持 thinking 的模型可设为 true
  system_prompt: ""

judge:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.0
  max_tokens: 4096
  use_llm_judge: false           # ToMQA 等开放式问答数据集需要设为 true

repeats: 3       # 重复轮数（本地推理建议 3，云端 API 建议 1）
max_samples: 0   # 0=全量，>0=随机抽样（调试时用 3-10）
datasets_path: datasets
results_path: results
```

### 3. 运行评测

```bash
# 运行所有数据集（约 10 个）
python run_all.py

# 运行单个数据集
python tasks/ToMBench/run.py
```

### 4. 生成结果表格

```bash
# 从 results/ 生成各数据集详细表格
python report/generate_dataset_tables.py

# 生成跨模型、跨数据集的汇总表
python report/generate_summary.py

# 查看汇总
cat tables/SUMMARY.md
```

### 5. Bad Case 分析

```bash
# 编辑 report/report_config.yaml 后运行
python report/report_client.py
```

---

## 目录结构

```
TomTest/
├── experiment_config.yaml        # 全局实验配置（模型 API、轮数、样本数等）
├── run_all.py                    # 全量运行入口（运行所有注册的数据集）
├── requirements.txt              # Python 依赖
│
├── tasks/                        # 各数据集评测代码
│   ├── ToMBench/
│   │   ├── config.yaml           # 数据集配置（path、schema、method）
│   │   ├── prompts.py            # build_prompt(row, method) → str
│   │   ├── metrics.py            # compute_metrics(predictions, data, judge) → dict
│   │   └── run.py                # 评测主脚本
│   ├── SocialIQA/
│   ├── ToMChallenges/
│   └── ...（共 12 个任务目录）
│
├── src/                          # 核心框架代码
│   ├── schemas.py                # Pydantic 输出 Schema（MCQAnswer, OpenAnswer 等）
│   ├── runner.py                 # 公共 runner 工具（加载配置、保存结果等）
│   ├── utils.py                  # 指标计算工具（compute_sample_metrics 等）
│   ├── llm/
│   │   ├── client.py             # LLMClient 基类（usage 统计、OpenAI 初始化）
│   │   ├── content_client.py     # ContentClient（自由文本生成）
│   │   ├── structure_client.py   # StructureClient（结构化输出，含自动降级）
│   │   └── llm_utils.py          # extract_json、format_schema_for_prompt 等
│   └── dataloader/
│       └── dataloader.py         # Arrow 格式数据集加载器
│
├── datasets/                     # Arrow 格式数据集（共 20+ 个）
│   ├── Belief_R/, FictionalQA/, HellaSwag/, RecToM/
│   ├── SimpleToM/, SocialIQA/, Tomato/, ToMBench/
│   ├── ToMChallenges/, ToMi/, ToMQA/, EmoBench/
│   └── ...
│
├── datasets_examples/            # 各数据集首条样本 JSON（快速了解字段格式）
│
├── results/                      # 评测输出（自动生成）
│   └── {dataset}/{model}/exp_{timestamp}/
│       ├── config.json           # 完整实验配置（API key 已脱敏）
│       ├── metrics.json          # 指标（avg_metrics + all_metrics）
│       └── prediction.jsonl      # 每条样本的预测详情
│
├── tables/                       # 结果表格（generate_dataset_tables.py 生成）
│   ├── SUMMARY.md                # 跨数据集 × 模型 accuracy 汇总
│   └── {dataset}/
│       ├── 基础指标.md            # accuracy、correct、total
│       └── 其他指标.md            # 细粒度指标（如 by_ability）
│
├── report/                       # 报告生成工具
│   ├── generate_dataset_tables.py  # 生成各数据集表格
│   ├── generate_summary.py         # 生成汇总表格
│   ├── report_client.py            # Bad Case 分析 + LLM 诊断报告
│   ├── tables_config.yaml          # generate_dataset_tables.py 配置
│   └── report_config.yaml          # report_client.py 配置
│
├── analysis/                     # Bad Case 分析报告（report_client.py 生成）
│   └── {dataset}/{model}/{timestamp}.md
│
├── docs/                         # 详细操作指南
│   ├── add_new_dataset.md        # 如何新增数据集
│   ├── add_new_model.md          # 如何用新模型测试
│   ├── generate_tables.md        # 如何生成数据集表格
│   ├── generate_summary.md       # 如何生成汇总表格
│   └── bad_case_analysis.md      # 如何做 Bad Case 分析
│
└── tests/
    └── test_dataloader.py        # 数据集加载验证
```

---

## 支持的数据集（当前启用 10 个）

| 数据集 | Schema | 说明 |
|---|---|---|
| Belief_R | `MCQAnswer3Lower` | 信念追踪（a/b/c 三选一） |
| FictionalQA | `MCQAnswer` | 虚构故事知识问答（四选一） |
| HellaSwag | `MCQAnswer` | 常识推理（四选一） |
| RecToM | `MultiLabelAnswer` | 推荐场景下的心智理论（多选） |
| SimpleTom | `MCQAnswer` | 简单 ToM 故事理解 |
| SocialIQA | `MCQAnswer3` | 社交情境理解（A/B/C 三选一） |
| Tomato | `MCQAnswer` | 多心智状态 MCQ（含选项随机打乱） |
| ToMBench | `MCQAnswer` | ToM 综合基准（中英文双语） |
| ToMChallenges | `MCQAnswer` | Anne-Sally & Smarties 经典测试 |
| ToMQA | `OpenAnswer` | 开放式 ToM 问答（LLM Judge） |

---

## 可用的 Schema（src/schemas.py）

| Schema | 字段 | 适用场景 |
|---|---|---|
| `MCQAnswer` | `answer: Literal["A","B","C","D"]` | 标准四选一 MCQ |
| `MCQAnswer2` | `answer: Literal["A","B"]` | 二选一 MCQ |
| `MCQAnswer3` | `answer: Literal["A","B","C"]` | 三选一 MCQ（大写） |
| `MCQAnswer3Lower` | `answer: Literal["a","b","c"]` | 三选一 MCQ（小写） |
| `OpenAnswer` | `answer: str` | 开放式问答 |
| `OneWordAnswer` | `answer: str`（无空白） | 单词/短语回答 |
| `MultiLabelAnswer` | `answer: List[str]`（单大写字母） | 多标签多选 |
| `JudgeAnswer` | `answer: Literal["True","False"]` | LLM Judge 判断 |

---

## 完整工作流

```
Step 1  配置 experiment_config.yaml（填写模型 API）
   ↓
Step 2  运行评测：python run_all.py
   ↓
Step 3  生成表格：python report/generate_dataset_tables.py
   ↓
Step 4  生成汇总：python report/generate_summary.py
   ↓
Step 5  Bad Case 分析：python report/report_client.py
```

---

## 操作指南

| 任务 | 文档 |
|---|---|
| 新增一个数据集进行评测 | [docs/add_new_dataset.md](docs/add_new_dataset.md) |
| 使用新模型（本地/云端）测试 | [docs/add_new_model.md](docs/add_new_model.md) |
| 从评测结果生成数据集对比表格 | [docs/generate_tables.md](docs/generate_tables.md) |
| 生成跨模型、跨数据集的汇总表 | [docs/generate_summary.md](docs/generate_summary.md) |
| 分析模型错误案例（Bad Case） | [docs/bad_case_analysis.md](docs/bad_case_analysis.md) |

---

## 更新日志

### 2026-04-22

- **新增 `MCQAnswer2`**：二选一 Schema（`Literal["A","B"]`），已注册到 `load_schema`
- **Schema 字段加入 `description`**：所有字段通过 `Field(description=...)` 标注语义
- **`MultiLabelAnswer` 严格化**：验证器只保留单个大写字母（A-Z），过滤多字符 token
- **`load_schema` 支持列表**：传入列表时返回 `{name: class}` 字典，支持 `config.yaml` 中 `schema: [...]` 写法
- **`--experiment-config` 参数**：所有 `tasks/*/run.py` 及 `run_all.py` 支持指定 experiment config 路径，默认 `experiment_config.yaml`，`run_all.py` 自动透传给子进程

---

### 2026-04-22（一）

- 新增 9 个数据集评测任务（Belief_R / FictionalQA / FollowBench / HellaSwag / IFEval / RecToM / SimpleTom / SocialIQA / ToMChallenges）
- LLM 客户端拆分重构：`client.py` → `client` / `content_client` / `structure_client` / `llm_utils` 四模块
- `src/runner.py`：新增 `sample_metas`、数据集级 `use_llm_judge` 覆盖、统一时间戳通过环境变量传递
- `src/schemas.py`：统一定义各任务 Schema，各任务不再维护独立 `schemas.py`
- 报告工具迁移至 `report/`，新增 `report_client.py`（Bad Case 分析）
- 新增文档：`docs/generate_tables.md`、`docs/generate_summary.md`、`docs/bad_case_analysis.md`、`report/README.md`

---

## 许可证

MIT License
