# eval — RAG 评估套件

两个层次，共用同一份数据集：

| 层次 | 命令 | 测什么 |
| --- | --- | --- |
| **检索** | `python -m eval.run_eval` | 检索质量（recall@k 等）+ 检索延迟 |
| **端到端** | `python -m eval.run_agent_eval` | 真实 agent 的答案质量 + TTFT/总延迟 + 成本 + 引用准确性 |

> 实测结果见 [REPORT.md](./REPORT.md)（东华大学研究生信息 KB，50 题）。

检索评估复刻生产链路 `KBSearchTool` 的检索流程（`embed → hybrid/dense search → rerank`），
但保留完整的有序命中列表，因此可以正确计算 recall 等排序指标。

---

# 检索评估

## 快速开始

在项目根目录，使用 backend 的虚拟环境运行：

```bash
backend\.venv\Scripts\python.exe -m eval.run_eval
```

默认评估 KB `c5a60b1d-c4a2-475a-b921-1242d93f51c2`（东华大学研究生信息）。

首次运行会从该 KB 的 chunk 中采样并自动生成评测集（默认用 LLM 出题），
缓存到 `eval/datasets/<kb_id>.jsonl`；结果打印为 markdown 报表并写入
`eval/results/<kb_id>_<时间戳>.json`。

## 常用参数

```bash
# 指定 KB、返回 top-5、指标看 @1/@3/@5、每题跑 3 次取延迟分位
python -m eval.run_eval --kb <kb_id> --limit 5 --top-k 1,3,5 --repeat 3

# 完全离线（不依赖 LLM key）生成评测集
python -m eval.run_eval --gen extractive --rebuild-dataset

# 用自己标注的数据集
python -m eval.run_eval --dataset path/to/my_dataset.jsonl
```

| 参数 | 说明 |
| --- | --- |
| `--kb` | 知识库 id |
| `--dataset` | 数据集 JSONL 路径（默认 `eval/datasets/<kb>.jsonl`） |
| `--limit` | 每题返回的 top-k（默认 5） |
| `--top-k` | 计算指标的 k 列表，逗号分隔（默认由 limit 推导） |
| `--repeat` | 每题重复次数，用于让延迟分位更稳（默认 1） |
| `--gen` | 出题方式：`llm` / `extractive`（默认 llm） |
| `--n` | 自动构造的题目数量（默认 30） |
| `--seed` | 采样随机种子 |
| `--rebuild-dataset` | 强制重建数据集 |
| `--out` | 结果 JSON 输出路径 |

## 指标

- **质量**（文档级、二值相关）：`recall@k`、`precision@k`、`hit@k`、`nDCG@k`、`MRR`
- **延迟**（毫秒）：`embed` / `search` / `rerank` / `total` 各自的
  mean、p50、p90、p95、max

## 自定义数据集格式

每行一个 JSON 对象：

```json
{"query": "研究生学籍异动需要哪些材料？", "relevant": ["东华大学研究生学籍管理规定（东华研〔2022〕13号）.pdf"], "meta": {}}
```

`relevant` 填**源文件名**（与检索结果里的 `filename` 一致）。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `metrics.py` | 质量指标 + 延迟分位（纯函数） |
| `retrieval.py` | 复刻 KBSearchTool 检索管线 + chunk 采样 |
| `dataset.py` | 数据集读写 + 自动出题（LLM / 抽取式） |
| `run_eval.py` | 检索评估 CLI、报表、结果落盘 |
| `agent_eval.py` | 进程内驱动真实 LangGraph agent（复刻 `app.py` 的 cfg 解析） |
| `judge.py` | LLM 评委：忠实度 / 正确性打分 |
| `run_agent_eval.py` | 端到端评估 CLI、报表、结果落盘 |

---

# agent 端到端评估

评估**完整 agent 链路**（`retrieve → plan → 流式生成`）的答案质量、端到端延迟与成本。

```bash
backend\.venv\Scripts\python.exe -m eval.run_agent_eval
```

与检索评估**共用同一份数据集**（`eval/datasets/<kb_id>.jsonl`），因此可以直接看
「检索对了 → 答案是否也对」。数据集不存在时会自动构建。

## 常用参数

```bash
# 只跑前 10 题（快速冒烟）
python -m eval.run_agent_eval --max-items 10

# 换模型
python -m eval.run_agent_eval --model glm-4-flash
```

| 参数 | 说明 |
| --- | --- |
| `--kb` | 知识库 id |
| `--dataset` | 数据集 JSONL 路径 |
| `--max-items` | 只跑前 N 题（0 = 全部） |
| `--model` | 覆盖 agent 使用的模型（v3-M6 同款 `dataclasses.replace` 语义） |
| `--gen` / `--n` / `--seed` / `--rebuild-dataset` | 数据集自动构建参数（同检索评估） |
| `--out` | 结果 JSON 输出路径 |

## 实现方式

**进程内直调 LangGraph**，而非走 HTTP：

```python
graph, cost = build_graph(emit=emit, kb=kb, llm_cfg=..., embedding_cfg=..., reranker_cfg=...)
final = await graph.ainvoke({"messages": [{"role": "user", "content": q}], "iterations": 0, "tool_call_log": []})
```

HTTP 路由相比此路径只多了 JWT 鉴权、限流和 SSE 封帧，三者都不改变被测指标；
不起服务、不登录使评测更快更稳。配置解析严格对齐
`app.py::_run_chat_session`（KB 级 embedding/reranker 优先 + 用户级 LLM + 环境兜底），
因此 agent 行为与线上一致。

延迟通过对 **emit 事件流打时间戳** 得到，映射真实用户体感：

```
t0 ──tool_end(search_kb)──► 首个 token ──────────► 结束
     └── 检索段 ──┘        └──── 生成段 ────┘
     └────────────── 总耗时 ───────────────┘
```

> 注：每轮 `build_graph` 都会新建 `CostTracker`，所以 `cost_usd` / tokens 是**单轮**口径，
> 与一个 HTTP 请求一致。评测不传入 `conversation_id` / `user_id`，即关闭 L1/L2/L4
> 记忆层，以保证可复现（记忆层不影响答案质量结论，但会引入 Redis/PG 依赖）。

## 指标

- **答案质量**（LLM 评委，1-5 归一化到 0-1）：`faithfulness` 忠实度、`correctness` 正确性、正确率
- **引用准确性**：引用命中率（`sources` 是否含标准答案文档）、引用 MRR、平均引用数
- **延迟**（毫秒）：首 token(TTFT) / 检索段 / 生成段 / 总耗时，各 mean、p50、p90、p95、max
- **成本**：总 `cost_usd`、单轮均值、in/out tokens
- **可靠性**：agent 失败数、工具失败数、评委失败数