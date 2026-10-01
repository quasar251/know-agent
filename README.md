# know — 私有 RAG 知识库 + 透明 Agent

> 上传文档 / 抓取网页 → 选中 KB → 用一句话问出来。
> 30 秒内吐一份带原文引用的 markdown 报告，全过程思考链可视。

**版本**：**v3.3-memory** · **线上**：https://know.cc.cd · **协议**：MIT

---

## 一、项目简介

know 是一个**私有 RAG 知识库 + 透明 Agent**：上传 md / txt / pdf / docx 或抓取网页 URL → 后台异步 ingest 建库 → 对话顶部选中该 KB 提问，Agent 会调用检索工具并把思考链实时展示给你，最后输出一份带原文引用的 markdown 报告。

核心能力：

- **私有 RAG 知识库**：4 种文档格式 + 网页抓取，后台异步 chunk / embed / upsert，支持重建索引。
- **多账户隔离与协作**：本地 JWT，每用户自建 KB；可邮箱邀请协作者（owner / editor / viewer）或生成匿名分享链接（可设过期、次数、撤销）。
- **混合检索**：Milvus 服务端 BM25 全文检索 + 稠密向量 RRF 融合；每个 KB 独立配 embedding（必填）+ 可选 cross-encoder reranker，互不干扰。
- **透明 Agent**：前端实时展示每一步工具调用 / 耗时 / 命中数；SSE 流式吐 token，三段式过渡（正在思考 → 工具跑 → 正在撰写报告）。
- **结构化报告 + 分享卡**：旅行 / 通用两套 skill 模板（TL;DR + sections + citations）；报告一键导出品牌图文 PNG。
- **双层记忆（v3.3）**：Redis 短期记忆（滑动窗口 + 批压缩摘要）+ 长期记忆（关键词 / 会话结束抽取 → PG + 向量召回）+ 用户画像；7 层上下文按 CJK 加权 token 预算注入，长对话不爆 context，且 chat 主流程永不因记忆系统 500。
- **会话级模型切换 & BYOK**：每个对话独立保存 LLM model；`BYOK_REQUIRED=true` 时强制用户自带 API key，杜绝公网白嫖。
- **后台管理**：`/admin` 统计看板 + 用户管理（封禁 / 设管理员 / 重置密码 / 删除）+ 跨用户 KB 管理，仅看元数据不看正文。
- **完全解耦**：LLM / embedding / 向量库 / App DB 全部 env 驱动或前端 `/settings` 可改，业务代码零改动。

---

## 二、整体架构

```
浏览器
  │ HTTPS / SSE
  ▼
┌────────────────────────────────┐
│  Next.js 14 (App Router)       │  :3000
│  - 聊天界面（流式 + 思考链）   │
│  - KB 管理 / 协作邀请          │
│  - 系统设置 / 记忆管理         │
│  - 内置 API proxy 转发到后端   │
└──────────────┬─────────────────┘
               │ proxy
               ▼
┌────────────────────────────────┐
│  FastAPI + LangGraph           │  :8000
│  ├─ Auth (JWT, bcrypt)         │
│  ├─ Conversations REST         │
│  ├─ KB REST (含协作 / 邀请)    │
│  ├─ Settings REST              │
│  ├─ Memories REST (v3.3)       │
│  └─ /api/chat (SSE)            │
│       └─ Agent 主循环          │
│            7 层上下文 + 双层记忆注入
│            plan → call_tools → skill_report
└──┬──────────┬──────────┬──────┘
   │          │          │
   ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────────────────┐
│ App DB   │ │ Memory   │ │ Vector DB            │
│ SQLite/PG│ │ Redis    │ │ Milvus Lite (默认)   │
│ users/kbs│ │ (v3.3)   │ │  或 Standalone/Qdrant│
│ msgs/    │ │ 短期窗口+ │ │ KB chunks +          │
│ memories │ │ 摘要+画像 │ │ 用户记忆向量 (v3.3)  │
└──────────┘ └──────────┘ └──────────────────────┘

外部服务（按 KB / 用户配置）：
  - LLM        : DeepSeek / Claude / OpenAI / SiliconFlow / Ollama
  - Embedding  : SiliconFlow BGE-M3 / OpenAI / Ollama
  - Reranker   : SiliconFlow / Cohere / 自托管 TEI (opt-in)
```

Agent 按所选 KB 切换工具集：用户 KB 只挂 `search_kb`（KB 模式先自动检索一次再生成）；通用聊天无工具纯直答；系统旅行示例 KB 挂旅行四件套（weather / restaurant_kb / amap / travel_report）。

---

## 三、项目目录

> 只列后端核心文件；前端与文档目录从略。

```
backend/                       Python 3.11 / FastAPI
├── src/
│   ├── app.py                 FastAPI 入口 + SSE chat 端点 + lifespan
│   ├── settings.py            pydantic-settings（.env 驱动）
│   ├── auth/                  JWT 认证（注册 / 登录 / me / 改密 / 删号）
│   ├── admin/                 后台管理 API（统计 / 用户 / KB）
│   ├── kb/                    知识库 / 文档 / 后台 ingest / 4 种格式解析器
│   ├── conversations/         会话历史 + v3.3 双层记忆子系统
│   ├── settings_user/         每用户自助配置（LLM / Embedding / Reranker）
│   ├── agent/                 LangGraph 主循环 + 7 层上下文构建
│   ├── tools/                 Agent 工具（kb_search / 旅行四件套）
│   ├── skills/                结构化报告模板（general / travel）
│   ├── safety/                输入清洗 / 输出脱敏 / 工具守卫
│   └── infra/                 解耦核心层（DB / 向量库 / Embedding / Reranker / LLM）
├── tests/                     测试
├── env.example                env 模板（每字段含注释）
└── pyproject.toml             依赖（含 dev / milvus / ollama 等 extras）
```

---

## 四、技术栈

**后端**

- Python 3.11 · FastAPI · LangGraph（Agent 主循环）
- SQLAlchemy 2.x async · pydantic-settings（env 驱动）
- 认证：bcrypt（cost=12）+ JWT HS256 · Fernet at-rest 加密 api_key
- 文档解析：pymupdf（PDF）· trafilatura（网页）· python-docx（docx）· markdown
- 流式：`sse-starlette` EventSourceResponse

**前端**

- Next.js 14（App Router）· React 18 · TypeScript 5.5
- Tailwind CSS 3.4（语义化 token + dark class 策略）
- react-markdown + remark-gfm（报告渲染）· lucide-react · sonner
- html2canvas + html2pdf（报告导出 / 品牌分享卡）
- SSE：`fetch` + ReadableStream 手动解析（支持 POST + Bearer）

**数据与存储**

- App DB：SQLite + aiosqlite（本地开发）/ PostgreSQL 16（生产）
- 向量库：Milvus Lite（默认嵌入式）/ Milvus Standalone / Zilliz Cloud / Qdrant
- 记忆热存储：Redis 7（v3.3，AOF 持久化）

**部署**

- Docker Compose 5 服务：`postgres:16-alpine` + `redis:7-alpine` + backend（自建）+ frontend（自建）+ `nginx:1.27-alpine`
- nginx 反代，`/api/chat` 关闭 buffering 以透传 SSE
- 镜像：backend Python 3.11 slim；frontend Next.js standalone（multi-stage，~150MB）

---

## 五、部署方式

### 方式一：Docker Compose（推荐，线上实例采用）

5 个容器组成整套栈，数据经 volume 持久化，业务代码零改动即可切换部件。

```bash
# 1. 准备 .env
cp env.docker.example .env
# 编辑填入：
#   POSTGRES_PASSWORD=$(openssl rand -hex 16)
#   JWT_SECRET=$(openssl rand -hex 32)
#   PUBLIC_URL=http://你的IP或域名

# 2. 一行起栈（build → up → 健康检查 → 打印状态）
./scripts/deploy.sh

# 3. 查看状态与日志
docker compose ps
./scripts/logs.sh backend
```

服务与数据卷：

| 服务 | 镜像 | 说明 |
|---|---|---|
| `postgres` | `postgres:16-alpine` | App DB → volume `know_postgres-data` |
| `redis` | `redis:7-alpine` | v3.3 记忆热存储 → volume `know_redis-data` |
| `backend` | 自建 | FastAPI + LangGraph + Milvus Lite 嵌入式 → volume `know_backend-data` |
| `frontend` | 自建 | Next.js 14 standalone build |
| `nginx` | `nginx:1.27-alpine` | 反代 :80，SSE-safe |

运维脚本：`scripts/deploy.sh`（构建 + 启动 + 健康检查）、`scripts/backup.sh`（备份 PG + backend-data 卷到 `./backups/`）、`scripts/logs.sh`（tail 服务日志）。

### 方式二：本地开发

前置：Python 3.11+ · Node.js 20+。

```bash
# 1. 后端
cd backend
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e '.[milvus]'        # 默认带 Milvus Lite
cp env.example .env               # 编辑填关键 key
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

# 2. 前端（新窗口）
cd frontend
npm install
npm run dev                       # :3000
```

Windows 下也可直接双击根目录 `start_local.bat` 一键启动前后端两个进程。

### 切换部件

App DB / 向量库 / embedding 全部由 env 决定，改 `DATABASE_URL`、`VECTOR_STORE`、`MILVUS_URI` / `QDRANT_URL` 等变量即可，业务代码无需改动。例如本地 SQLite → 生产 PostgreSQL：

```bash
DATABASE_URL=postgresql+asyncpg://know:password@postgres:5432/know
```

---

Powered by Claude / DeepSeek / FastAPI / LangGraph / Next.js / Milvus / SiliconFlow · MIT 协议