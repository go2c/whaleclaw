# Phase 1: 基座 — 配置 + Gateway + 基础 Agent 循环

## 目标

搭建 WhaleClaw 项目骨架。完成后应能:
1. 通过 CLI 启动 Gateway 服务
2. 客户端通过 WebSocket 连接 Gateway
3. 发送一条文本消息，经 Agent 调用 Anthropic Claude API，流式返回回复

## 前置条件

- Python 3.12 (项目内嵌 `./python/bin/python3.12`)
- Anthropic API Key (用于测试)

---

## 1. 项目初始化

### 1.1 pyproject.toml

```toml
[project]
name = "whaleclaw"
version = "0.1.0"
description = "WhaleClaw — Personal AI Assistant (Python)"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "websockets>=14.0",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "httpx>=0.28",
    "typer>=0.15",
    "rich>=13.9",
    "structlog>=24.4",
    "aiosqlite>=0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.8",
    "pyright>=1.1",
    "pytest-mock>=3.14",
]

[project.scripts]
whaleclaw = "whaleclaw.cli.main:app"

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM", "TCH"]

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 1.2 目录创建

创建以下目录和 `__init__.py`:

```
whaleclaw/__init__.py
whaleclaw/config/__init__.py
whaleclaw/gateway/__init__.py
whaleclaw/agent/__init__.py
whaleclaw/providers/__init__.py
whaleclaw/channels/__init__.py
whaleclaw/cli/__init__.py
whaleclaw/utils/__init__.py
tests/conftest.py
```

---

## 2. 配置系统

### 2.1 路径常量 — `whaleclaw/config/paths.py`

```python
WHALECLAW_HOME = Path.home() / ".whaleclaw"
CONFIG_FILE = WHALECLAW_HOME / "whaleclaw.json"
CREDENTIALS_DIR = WHALECLAW_HOME / "credentials"
SESSIONS_DIR = WHALECLAW_HOME / "sessions"
WORKSPACE_DIR = WHALECLAW_HOME / "workspace"
LOGS_DIR = WHALECLAW_HOME / "logs"
```

启动时自动创建不存在的目录。

### 2.2 配置 Schema — `whaleclaw/config/schema.py`

使用 Pydantic v2 定义配置模型:

```python
class GatewayConfig(BaseModel):
    port: int = 18666
    bind: str = "127.0.0.1"
    verbose: bool = False

class AgentConfig(BaseModel):
    model: str = "anthropic/claude-sonnet-4-20250514"
    workspace: str = str(WORKSPACE_DIR)

class WhaleclawConfig(BaseModel):
    gateway: GatewayConfig = GatewayConfig()
    agent: AgentConfig = AgentConfig()
```

### 2.3 配置加载 — `whaleclaw/config/loader.py`

- 从 JSON 文件加载配置
- 支持环境变量覆盖 (`WHALECLAW_GATEWAY_PORT` 等)
- 支持命令行参数覆盖
- 提供 `get_config() -> WhaleclawConfig` 全局访问点

---

## 3. 日志系统

### 3.1 structlog 配置 — `whaleclaw/utils/log.py`

- 开发模式: 彩色控制台输出 (ConsoleRenderer)
- 生产模式: JSON 格式输出到文件
- 日志级别: 通过 `--verbose` 控制 (INFO / DEBUG)
- 自动附加 timestamp、logger name

```python
def setup_logging(verbose: bool = False) -> None:
    """配置 structlog，verbose=True 时输出 DEBUG 级别"""
```

---

## 4. Gateway 核心

### 4.1 FastAPI 应用工厂 — `whaleclaw/gateway/app.py`

```python
def create_app(config: WhaleclawConfig) -> FastAPI:
    """创建 FastAPI 应用实例，注册路由和中间件"""
```

路由:
- `GET /` — 健康检查，返回 `{"status": "ok", "version": "0.1.0"}`
- `GET /api/status` — Gateway 状态信息
- `WS /ws` — WebSocket 端点

### 4.2 WebSocket 协议 — `whaleclaw/gateway/protocol.py`

定义消息类型枚举和 Pydantic 模型:

```python
class MessageType(str, Enum):
    MESSAGE = "message"
    STREAM = "stream"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

class WSMessage(BaseModel):
    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None
    payload: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 4.3 WebSocket 处理 — `whaleclaw/gateway/ws.py`

```python
async def websocket_handler(websocket: WebSocket, config: WhaleclawConfig) -> None:
    """
    WebSocket 连接生命周期:
    1. 接受连接
    2. 心跳循环 (30s ping/pong)
    3. 消息接收 -> 分发到 Agent
    4. Agent 回复 -> 流式推送到客户端
    5. 断开清理
    """
```

关键行为:
- 连接时分配默认 session_id (如果客户端未提供)
- 收到 `ping` 回复 `pong`
- 收到 `message` 类型时，提取 `payload.content` 交给 Agent 处理
- Agent 回复通过 `stream` 类型逐块推送，最后发一条完整的 `message`

---

## 5. 基础 Agent 循环

### 5.1 Agent 循环 — `whaleclaw/agent/loop.py`

```python
async def run_agent(
    message: str,
    session_id: str,
    config: WhaleclawConfig,
    on_stream: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """
    基础 Agent 循环 (Phase 1 单轮):
    1. PromptAssembler 按预算组装系统提示词 (静态层)
    2. 组装消息列表 [system, user]
    3. 调用 LLM Provider (流式)
    4. 收集完整回复并返回
    """
```

### 5.2 提示词构建 — `whaleclaw/agent/prompt.py`

**设计原则**: 系统提示词分层按需组装，避免 OpenClaw 式全量注入导致的 token 浪费。

```python
class PromptLayer(str, Enum):
    """提示词层级"""
    STATIC = "static"       # 固定身份 + 核心规则 (~200 tokens，每轮必带)
    DYNAMIC = "dynamic"     # 按需注入 (技能路由/记忆检索，预算 60%)
    LAZY = "lazy"           # 条件触发 (AGENTS.md 摘要/EvoMap，预算剩余)

class PromptAssembler:
    """分层提示词组装器 — 在 token 预算内按优先级组装"""

    def build(
        self,
        config: WhaleclawConfig,
        user_message: str,
        session: Session | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        """
        按预算组装系统提示词:
        1. 静态层 (必选, ~200 tokens): 身份 + 核心行为规则
        2. 动态层 (按需, 预算 60%): 技能路由结果 + 记忆检索
        3. 延迟层 (条件触发, 预算剩余): AGENTS.md 摘要 / EvoMap 方案

        token_budget 默认根据模型 max_context 自动计算:
        budget = max_context * 0.15  (为用户消息和回复留足空间)
        """

    def _build_static(self, config: WhaleclawConfig) -> str:
        """构建静态层 — 极简身份和核心规则"""

    def _build_dynamic(
        self, user_message: str, session: Session | None, budget: int,
    ) -> str:
        """构建动态层 — 技能路由 + 记忆检索 (Phase 5/7 实现)"""

    def _build_lazy(self, session: Session | None, budget: int) -> str:
        """构建延迟层 — 仅首轮或显式触发 (Phase 5 实现)"""

    def estimate_tokens(self, text: str) -> int:
        """快速 token 估算 (中文 ~1.5 char/token, 英文 ~4 char/token)"""
```

Phase 1 默认静态层 (~200 tokens):
```
你是 WhaleClaw，一个个人 AI 助手。
- 使用用户的语言回复
- 简洁准确，不废话
- 不确定时坦诚说明，不编造信息
- 工具调用遵循提供的 JSON Schema
```

**与 OpenClaw 的关键区别**: OpenClaw 将 AGENTS.md + TOOLS.md + 全部 SKILL.md + 记忆全量塞入 system prompt (约 8000+ tokens/轮)。WhaleClaw 的 PromptAssembler 通过分层 + 预算控制，将每轮 system prompt 控制在 ~1200 tokens，节省约 85% 的输入 token。

Phase 1 阶段只实现静态层，动态层和延迟层在后续阶段渐进实现。

---

## 6. LLM Provider — Anthropic

### 6.1 Provider 基类 — `whaleclaw/providers/base.py`

```python
class CacheControl(BaseModel):
    """提示词缓存标记 (Anthropic prompt caching / Google context caching)"""
    type: Literal["ephemeral"] = "ephemeral"

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    cache_control: CacheControl | None = None   # 标记为可缓存 (静态层自动标记)

class ToolSchema(BaseModel):
    """工具 JSON Schema — 传递给 LLM 原生 tools 参数"""
    name: str
    description: str
    input_schema: dict[str, Any]

class LLMProvider(ABC):
    supports_native_tools: bool = True           # 是否支持原生 tools 参数
    supports_cache_control: bool = False         # 是否支持 prompt caching

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,  # 工具走原生参数，不塞 system prompt
        on_stream: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResponse:
        """
        发送消息列表，返回完整回复。支持流式回调。
        tools 通过 LLM API 原生 tools 参数传递 (不占 system prompt token)。
        对不支持原生 tools 的 Provider，自动降级为 system prompt 注入。
        """
```

**工具描述不再注入 system prompt**: 所有主流 LLM 都支持原生 `tools` 参数，工具的 JSON Schema 通过该参数传递。好处:
- 不占 system prompt token 预算
- Anthropic/Google 对 tools 参数有缓存优化，多轮对话只计费一次
- 结构化传递比纯文本更准确
- 对不支持原生 tools 的 Provider (如部分 NVIDIA NIM 免费模型)，Provider 层自动降级为 system prompt 注入

### 6.2 Anthropic 适配器 — `whaleclaw/providers/anthropic.py`

- 使用 `httpx` 直接调用 Anthropic Messages API
- 支持流式响应 (`stream=True`)
- API Key 从配置/环境变量 `ANTHROPIC_API_KEY` 读取
- 错误处理: 速率限制重试、API 错误映射

```python
class AnthropicProvider(LLMProvider):
    BASE_URL = "https://api.anthropic.com/v1/messages"
    supports_native_tools = True
    supports_cache_control = True                # Anthropic 支持 prompt caching

    async def chat(self, messages, model, tools=None, on_stream=None) -> AgentResponse:
        """
        调用 Anthropic Messages API，流式返回。
        - tools 通过 API 原生 tools 参数传递
        - 静态层 system prompt 标记 cache_control 以启用 prompt caching
        """
```

---

## 7. CLI 骨架

### 7.1 CLI 入口 — `whaleclaw/cli/main.py`

```python
app = typer.Typer(name="whaleclaw", help="WhaleClaw — Personal AI Assistant")
```

### 7.2 Gateway 命令 — `whaleclaw/cli/gateway_cmd.py`

```python
@app.command()
def run(
    port: int = typer.Option(18666, help="Gateway 端口"),
    bind: str = typer.Option("127.0.0.1", help="绑定地址"),
    verbose: bool = typer.Option(False, help="详细日志"),
) -> None:
    """启动 WhaleClaw Gateway"""
```

执行流程:
1. 加载配置 (文件 + 环境变量 + CLI 参数)
2. 初始化日志
3. 创建 FastAPI 应用
4. 启动 uvicorn

---

## 8. 版本管理

### 8.1 版本文件 — `whaleclaw/version.py`

```python
__version__ = "0.1.0"
```

在 `whaleclaw/__init__.py` 中导出:
```python
from whaleclaw.version import __version__
```

---

## 验收标准

完成 Phase 1 后，以下场景必须通过:

### AC-1: Gateway 启动
```bash
./python/bin/python3.12 -m whaleclaw gateway run --port 18666 --verbose
# 输出: Gateway 启动日志，监听 127.0.0.1:18666
```

### AC-2: 健康检查
```bash
curl http://127.0.0.1:18666/
# 返回: {"status": "ok", "version": "0.1.0"}
```

### AC-3: WebSocket 连接 + 消息
使用 `websocat` 或 Python 脚本:
```python
import asyncio, websockets, json

async def test():
    async with websockets.connect("ws://127.0.0.1:18666/ws") as ws:
        msg = {"type": "message", "payload": {"content": "你好"}}
        await ws.send(json.dumps(msg))
        # 应收到多条 stream 类型消息，最后一条 message 类型
        async for response in ws:
            data = json.loads(response)
            print(data["type"], data.get("payload", {}).get("content", ""))
            if data["type"] == "message":
                break

asyncio.run(test())
```

### AC-4: 配置加载
```bash
# 创建配置文件
mkdir -p ~/.whaleclaw
echo '{"gateway": {"port": 19000}}' > ~/.whaleclaw/whaleclaw.json
./python/bin/python3.12 -m whaleclaw gateway run
# 应监听 19000 端口
```

### AC-5: 单元测试
```bash
./python/bin/python3.12 -m pytest tests/ -v
# 所有测试通过
```

需要的测试:
- `tests/test_config/test_loader.py` — 配置加载/合并/默认值
- `tests/test_config/test_schema.py` — 配置 schema 校验
- `tests/test_gateway/test_protocol.py` — WS 消息序列化/反序列化
- `tests/test_gateway/test_app.py` — 健康检查端点
- `tests/test_providers/test_anthropic.py` — Anthropic Provider (mock HTTP)
- `tests/test_agent/test_loop.py` — Agent 循环 (mock Provider)

---

## 文件清单

本阶段需要创建的所有文件:

```
pyproject.toml
whaleclaw/__init__.py
whaleclaw/version.py
whaleclaw/types.py
whaleclaw/entry.py
whaleclaw/config/__init__.py
whaleclaw/config/paths.py
whaleclaw/config/schema.py
whaleclaw/config/loader.py
whaleclaw/gateway/__init__.py
whaleclaw/gateway/app.py
whaleclaw/gateway/protocol.py
whaleclaw/gateway/ws.py
whaleclaw/agent/__init__.py
whaleclaw/agent/loop.py
whaleclaw/agent/prompt.py
whaleclaw/providers/__init__.py
whaleclaw/providers/base.py
whaleclaw/providers/anthropic.py
whaleclaw/cli/__init__.py
whaleclaw/cli/main.py
whaleclaw/cli/gateway_cmd.py
whaleclaw/utils/__init__.py
whaleclaw/utils/log.py
tests/conftest.py
tests/test_config/test_loader.py
tests/test_config/test_schema.py
tests/test_gateway/test_protocol.py
tests/test_gateway/test_app.py
tests/test_providers/test_anthropic.py
tests/test_agent/test_loop.py
```
