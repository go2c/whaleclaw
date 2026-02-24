# Phase 7: 记忆 + 高级功能

## 目标

实现长期记忆和高级 Agent 能力:
1. 记忆系统 — 向量存储 + 自动摘要 + 上下文压缩
2. 定时任务 — Cron 调度器，定时消息/提醒
3. 媒体理解 — 图片识别、音频转文字、链接摘要
4. TTS/语音 — 文字转语音输出
5. Thinking 模式 — 可控的推理深度
6. 使用统计 — Token 用量追踪和成本估算

## 前置条件

- Phase 1 ~ 6 全部完成

---

## 1. 记忆系统

### 1.1 记忆抽象 — `whaleclaw/memory/base.py`

```python
class MemoryEntry(BaseModel):
    id: str
    content: str                     # 记忆内容
    source: str                      # 来源 (session_id, 手动添加等)
    tags: list[str] = []             # 标签
    importance: float = 0.5          # 重要性 (0-1)
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    embedding: list[float] | None = None

class MemoryStore(ABC):
    """记忆存储抽象基类"""

    @abstractmethod
    async def add(self, content: str, source: str, tags: list[str] = []) -> MemoryEntry: ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5, min_score: float = 0.5) -> list[MemorySearchResult]: ...

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None: ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    async def list_recent(self, limit: int = 20) -> list[MemoryEntry]: ...

class MemorySearchResult(BaseModel):
    entry: MemoryEntry
    score: float                     # 相似度分数 (0-1)
```

### 1.2 当前实现后端 — `whaleclaw/memory/vector.py`

当前默认实现为 `SimpleMemoryStore`：

- 存储: `~/.whaleclaw/memory/memory.json`
- 检索: 关键词匹配 + 分数阈值
- 管理: `MemoryManager` 中做去重、限频、批量写入、重排与裁剪

说明:
- 设计上保留 `MemoryStore` 抽象，可后续切换到向量库（如 ChromaDB/LanceDB）。

### 1.3 自动摘要 — `whaleclaw/memory/summary.py`

```python
class ConversationSummarizer:
    """对话摘要生成器"""

    async def summarize(self, messages: list[Message], provider: LLMProvider) -> str:
        """将消息列表压缩为摘要"""

    async def extract_facts(self, messages: list[Message], provider: LLMProvider) -> list[str]:
        """从对话中提取关键事实 (用于长期记忆)"""
```

### 1.4 自动记忆管理

Agent 循环中集成记忆:

```python
class MemoryManager:
    """记忆管理器 — 集成到 Agent 循环 (带 token 预算控制)"""

    async def recall(self, query: str, max_tokens: int = 500) -> str:
        """
        根据当前对话检索相关记忆，在 max_tokens 预算内格式化:
        1. 向量搜索 top-K 条 (K=10)
        2. 按相关性分数排序
        3. 逐条累加 token 估算，超预算则截断
        4. 格式化为紧凑的记忆片段 (不含冗余前缀/分隔符)

        与 OpenClaw 的区别: OpenClaw 的记忆检索无上限，可能注入 1000+ tokens。
        WhaleClaw 通过 max_tokens 参数严格控制，由 PromptAssembler 的
        动态层预算决定 (默认 ~500 tokens)。
        """

    async def auto_capture_user_message(...) -> bool:
        """按规则自动捕获，支持去重/限频/缓冲批量写入"""

    async def organize_if_needed(...) -> bool:
        """低频调用 LLM 生成 L0/L1 画像，并裁剪旧原始记忆"""

    async def get_global_style_directive() -> str:
        """读取长期记忆提炼出的全局回复风格"""
```

记忆注入时机:
- `run_agent` 入口先按 `recall_policy` 判断是否触发 recall
- 创作类任务（PPT/文档/代码等）自动注入 L0 画像
- raw 记忆只在强意图（继续上次/按之前）时补充
- 全局风格指令（style_directive）可每轮低成本注入，用户本轮明确要求优先

### 1.5 全局风格 API

Gateway 提供:

```
GET    /api/memory/style
POST   /api/memory/style
DELETE /api/memory/style
```

用于 WebChat 设置面板查看/手动覆盖/清除全局回复风格。

### 1.6 EvoMap 知识集成

EvoMap 拉取的已验证资产 (Capsule) 可作为记忆来源:

```python
class EvoMapMemoryBridge:
    """将 EvoMap 资产桥接到记忆系统"""

    async def index_assets(self, assets: list[Asset]) -> None:
        """将 EvoMap Capsule 索引到向量存储，使 Agent 可通过记忆检索找到"""

    async def search_solutions(
        self, error_signals: list[str], max_tokens: int = 300,
    ) -> list[Asset]:
        """
        Agent 遇到错误时 (条件触发，不是每轮都注入):
        1. 先搜索本地记忆
        2. 再搜索 EvoMap 缓存的资产
        3. 如果本地无结果，实时调用 EvoMap fetch
        结果在 max_tokens 预算内格式化。
        """
```

**条件触发**: EvoMap 方案仅在 PromptAssembler 延迟层检测到错误信号时注入 (如工具执行失败、用户描述了 bug)，不会在每轮对话中浪费 token。

---

## 2. 定时任务

### 2.1 Cron 调度器 — `whaleclaw/cron/scheduler.py`

```python
class CronJob(BaseModel):
    id: str
    name: str
    schedule: str                    # cron 表达式 ("0 9 * * *")
    action: CronAction
    enabled: bool = True
    created_at: datetime
    last_run: datetime | None = None
    next_run: datetime | None = None

class CronAction(BaseModel):
    type: Literal["message", "agent", "webhook"]
    target: str                      # 目标 (session_id, URL 等)
    payload: dict[str, Any]          # 动作参数

class CronScheduler:
    """定时任务调度器"""

    async def start(self) -> None:
        """启动调度循环"""

    async def stop(self) -> None:
        """停止调度"""

    async def add_job(self, job: CronJob) -> None: ...
    async def remove_job(self, job_id: str) -> None: ...
    async def list_jobs(self) -> list[CronJob]: ...
    async def trigger_job(self, job_id: str) -> None:
        """手动触发"""
```

### 2.2 提醒工具

```python
class ReminderTool(Tool):
    """设置提醒"""
    # 参数: message (str), time (str, 如 "明天上午9点")
    # 实现: 解析时间 -> 创建 CronJob -> 到时发送消息
```

### 2.3 定时消息

Agent 可通过工具创建定时任务:
```
用户: 每天早上9点给我发天气预报
Agent: [调用 cron 工具: 创建定时任务]
Agent: 已设置每日 9:00 天气提醒
```

### 2.4 EvoMap 定时同步

EvoMap 插件 (Phase 5) 注册为内置 Cron 任务:

```python
EVOMAP_SYNC_JOB = CronJob(
    id="evomap-sync",
    name="EvoMap 网络同步",
    schedule="0 */4 * * *",              # 每 4 小时
    action=CronAction(
        type="agent",
        target="evomap",
        payload={"steps": ["hello", "fetch", "publish", "claim"]},
    ),
)
```

同步流程:
1. **hello** — 刷新节点注册，更新 reputation
2. **fetch** — 拉取新推广资产，缓存到 `~/.whaleclaw/evomap/assets/`
3. **publish** — 上传待发布的本地资产 (Agent 修复过的问题)
4. **claim** — 自动认领匹配的高价值赏金任务 (可配置)

同步结果写入记忆系统，Agent 可在后续对话中引用 EvoMap 获取的解决方案。

---

## 3. 媒体理解

### 3.1 媒体处理管道 — `whaleclaw/media/pipeline.py`

```python
class MediaPipeline:
    """媒体处理管道"""

    async def process(self, attachment: MediaAttachment) -> MediaResult:
        """
        根据媒体类型分发处理:
        - image -> VisionProcessor
        - audio -> TranscriptionProcessor
        - video -> 提取关键帧 + 音频转文字
        - file -> 文本提取 (PDF/DOCX/etc)
        """

class MediaResult(BaseModel):
    type: str
    text: str | None = None          # 提取的文本
    description: str | None = None   # 描述 (图片)
    metadata: dict[str, Any] = {}
```

### 3.2 图片理解 — `whaleclaw/media/vision.py`

```python
class VisionProcessor:
    """图片理解 (多模态模型)"""

    async def describe(self, image_path: str, prompt: str | None = None) -> str:
        """
        使用多模态模型描述图片:
        - Anthropic: Claude Vision
        - OpenAI: GPT-4o Vision
        - 通义千问: qwen-vl
        - 智谱: glm-5 / glm-4.7 (多模态)
        - 月之暗面: kimi-k2.5 (原生多模态，支持图片+视频)
        - Google: gemini-3.1-pro-preview / gemini-3-flash-preview (原生多模态)
        """

    async def ocr(self, image_path: str) -> str:
        """提取图片中的文字"""
```

### 3.3 音频转文字 — `whaleclaw/media/transcribe.py`

```python
class TranscriptionProcessor:
    """音频转文字"""

    async def transcribe(self, audio_path: str, language: str = "zh") -> str:
        """
        音频转文字:
        - OpenAI Whisper API
        - 或本地 whisper 模型
        """
```

### 3.4 链接摘要 — `whaleclaw/media/link_summary.py`

```python
class LinkSummarizer:
    """URL 链接内容摘要"""

    async def summarize(self, url: str) -> str:
        """
        1. 抓取网页内容
        2. 提取正文 (readability)
        3. 使用 LLM 生成摘要
        """
```

---

## 4. TTS / 语音

### 4.1 TTS 引擎 — `whaleclaw/tts/engine.py`

```python
class TTSEngine(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice: str = "default") -> bytes:
        """文字转语音，返回音频数据"""

class OpenAITTS(TTSEngine):
    """OpenAI TTS API"""
    # 支持: alloy, echo, fable, onyx, nova, shimmer

class EdgeTTS(TTSEngine):
    """Microsoft Edge TTS (免费)"""
    # 支持中文语音
```

### 4.2 语音配置

```json
{
  "tts": {
    "enabled": true,
    "engine": "edge",
    "voice": "zh-CN-XiaoxiaoNeural",
    "auto_tts": false
  }
}
```

---

## 5. Thinking 模式

### 5.1 思考深度控制

```python
class ThinkingLevel(str, Enum):
    OFF = "off"           # 不输出思考过程
    LOW = "low"           # 简短思考
    MEDIUM = "medium"     # 中等思考
    HIGH = "high"         # 深度思考
    XHIGH = "xhigh"      # 最深度思考 (extended thinking)

def apply_thinking_level(level: ThinkingLevel, provider_config: dict) -> dict:
    """
    根据思考深度调整 Provider 参数:
    - Anthropic: extended thinking budget
    - OpenAI: reasoning effort
    - DeepSeek: deepseek-reasoner 模型
    """
```

### 5.2 思考过程展示

- WebChat: 折叠面板显示思考过程
- 飞书: 卡片中可折叠的 "思考过程" 区域
- CLI: 灰色文字显示

---

## 6. 使用统计

### 6.1 Token 追踪 — `whaleclaw/agent/usage.py`

```python
class UsageTracker:
    """Token 用量和成本追踪"""

    async def record(self, session_id: str, usage: TokenUsage) -> None: ...
    async def get_session_usage(self, session_id: str) -> SessionUsage: ...
    async def get_daily_usage(self, date: str | None = None) -> DailyUsage: ...
    async def get_total_usage(self) -> TotalUsage: ...

class TokenUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    cost_usd: float                  # 估算成本

class SessionUsage(BaseModel):
    session_id: str
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    request_count: int
```

### 6.2 成本估算

内置各模型的定价表:

```python
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1m_tokens, output_per_1m_tokens) USD
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "deepseek-chat": (0.14, 0.28),
    "qwen-max": (2.4, 9.6),
    "glm-5": (2.0, 8.0),
    "glm-4.7": (1.0, 4.0),
    "glm-4.7-flash": (0.0, 0.0),         # 免费
    "MiniMax-M2.5": (1.0, 5.0),
    "MiniMax-M2.1": (1.0, 5.0),
    "kimi-k2.5": (0.6, 3.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3-pro-preview": (2.0, 12.0),
    "gemini-3-flash-preview": (0.1, 0.4),
    "meta/llama-3.1-8b-instruct": (0.0, 0.0),  # NVIDIA NIM 免费
}
```

### 6.3 /status 命令增强

```
/status
━━━━━━━━━━━━━━━━━━━━━━━━━━━
模型: anthropic/claude-sonnet-4-20250514
会话消息: 24 条
Token 用量: 12,345 输入 / 3,456 输出
估算成本: $0.089
思考深度: medium
记忆条目: 15
━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. 聊天命令扩展

新增命令:

| 命令 | 说明 |
|------|------|
| `/compact` | 压缩上下文 (摘要 + 存入记忆) |
| `/think <level>` | 设置思考深度 |
| `/usage` | 显示 Token 用量统计 |
| `/memory search <query>` | 搜索记忆 |
| `/memory list` | 列出最近记忆 |
| `/memory forget <id>` | 删除指定记忆 |
| `/remind <time> <message>` | 设置提醒 |
| `/tts on\|off` | 开关语音回复 |
| `/voice <voice_name>` | 切换语音 |
| `/evomap status` | EvoMap 节点状态 (reputation/积分) |
| `/evomap sync` | 手动触发 EvoMap 同步 |
| `/evomap search <signals>` | 搜索 EvoMap 已验证方案 |
| `/evomap tasks` | 查看可用赏金任务 |

---

## 验收标准

### AC-1: 长期记忆
```
会话1: 我最喜欢的编程语言是 Rust
(重置会话)
会话2: 我喜欢什么编程语言？
Agent: 根据我的记忆，你最喜欢的编程语言是 Rust。
```

### AC-2: 上下文压缩
长对话后执行 `/compact`:
- 历史消息被压缩为摘要
- 关键事实存入长期记忆
- 后续对话仍能引用之前的信息

### AC-3: 定时提醒
```
用户: 5分钟后提醒我喝水
Agent: 已设置提醒，5分钟后通知你。
(5分钟后)
Agent: 提醒: 该喝水了！
```

### AC-4: 图片理解
发送一张图片:
- Agent 能描述图片内容
- 支持追问图片细节

### AC-5: 音频转文字
发送一段语音消息:
- Agent 自动转为文字
- 基于文字内容回复

### AC-6: 使用统计
```
/usage
# 显示详细的 Token 用量和成本
```

### AC-7: Thinking 模式
```
/think high
用户: 分析这段代码的时间复杂度
Agent: (显示详细思考过程) 时间复杂度分析: ...
```

### AC-8: EvoMap 知识检索
Agent 遇到 `TimeoutError` 时，自动从 EvoMap 缓存中检索到相关 Capsule，并参考其解决方案修复问题。

### AC-9: EvoMap 定时同步
Cron 调度器每 4 小时自动执行 EvoMap 同步，拉取的新资产自动索引到记忆系统。

---

## 文件清单

```
whaleclaw/memory/__init__.py
whaleclaw/memory/base.py
whaleclaw/memory/vector.py
whaleclaw/memory/summary.py
whaleclaw/memory/manager.py
whaleclaw/cron/__init__.py
whaleclaw/cron/scheduler.py
whaleclaw/cron/store.py
whaleclaw/media/__init__.py
whaleclaw/media/pipeline.py
whaleclaw/media/vision.py
whaleclaw/media/transcribe.py
whaleclaw/media/link_summary.py
whaleclaw/tts/__init__.py
whaleclaw/tts/engine.py
whaleclaw/tts/openai_tts.py
whaleclaw/tts/edge_tts.py
whaleclaw/agent/usage.py
whaleclaw/agent/thinking.py
whaleclaw/tools/reminder.py
whaleclaw/tools/cron_tool.py
whaleclaw/tools/memory_tool.py
tests/test_memory/test_vector.py
tests/test_memory/test_summary.py
tests/test_memory/test_manager.py
tests/test_cron/test_scheduler.py
tests/test_media/test_pipeline.py
tests/test_media/test_vision.py
tests/test_media/test_transcribe.py
tests/test_tts/test_engine.py
tests/test_agent/test_usage.py
tests/test_agent/test_thinking.py
```
