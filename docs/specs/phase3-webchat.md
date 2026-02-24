# Phase 3: WebChat 渠道

## 目标

实现内置 Web 聊天界面，用户可通过浏览器与 WhaleClaw Agent 对话:
1. 现代化聊天 UI，支持 Markdown 渲染和代码高亮
2. 实时流式显示 Agent 回复和工具调用过程
3. 会话管理 (新建/切换/删除)
4. 基础认证 (Token 或密码)

## 前置条件

- Phase 1 + Phase 2 全部完成

---

## 1. 渠道抽象层

### 1.1 Channel 基类 — `whaleclaw/channels/base.py`

```python
class ChannelMessage(BaseModel):
    """渠道无关的标准消息格式"""
    id: str
    channel: str                     # "webchat", "feishu", ...
    peer_id: str                     # 发送者标识
    group_id: str | None = None      # 群组标识 (单聊为 None)
    content: str                     # 文本内容
    media: list[MediaAttachment] = []
    reply_to: str | None = None      # 回复的消息 ID
    timestamp: datetime
    raw: dict[str, Any] = {}         # 原始渠道消息

class MediaAttachment(BaseModel):
    type: Literal["image", "audio", "video", "file"]
    url: str | None = None
    path: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    size: int | None = None

class ChannelPlugin(ABC):
    """渠道插件抽象基类"""
    name: str

    @abstractmethod
    async def start(self) -> None:
        """启动渠道 (连接/注册 webhook 等)"""

    @abstractmethod
    async def stop(self) -> None:
        """停止渠道"""

    @abstractmethod
    async def send(self, peer_id: str, content: str, **kwargs) -> None:
        """发送消息到渠道"""

    async def send_stream(self, peer_id: str, stream: AsyncIterator[str]) -> None:
        """流式发送 (默认收集后一次发送，渠道可覆盖实现增量更新)"""

    @abstractmethod
    async def on_message(self, callback: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        """注册消息回调"""
```

### 1.2 渠道管理器 — `whaleclaw/channels/manager.py`

```python
class ChannelManager:
    """管理所有渠道的生命周期和消息路由"""

    def register(self, channel: ChannelPlugin) -> None: ...
    async def start_all(self) -> None: ...
    async def stop_all(self) -> None: ...
    async def broadcast(self, content: str) -> None: ...
```

---

## 2. WebChat 后端

### 2.1 WebChat 渠道 — `whaleclaw/channels/webchat/handler.py`

```python
class WebChatChannel(ChannelPlugin):
    name = "webchat"

    async def start(self) -> None:
        """注册到 Gateway 的 WebSocket 路由"""

    async def send(self, peer_id: str, content: str, **kwargs) -> None:
        """通过 WebSocket 发送消息到前端"""

    async def send_stream(self, peer_id: str, stream) -> None:
        """流式推送到前端 WebSocket"""
```

### 2.2 REST API 端点

Gateway 新增 REST 路由:

```
GET  /api/sessions                  # 获取会话列表
POST /api/sessions                  # 创建新会话
GET  /api/sessions/{id}             # 获取会话详情 (含消息历史)
DELETE /api/sessions/{id}           # 删除会话
POST /api/sessions/{id}/compact     # 压缩会话上下文
POST /api/auth/login                # 登录 (密码认证)
GET  /api/auth/verify               # 验证 Token
GET  /api/memory/style              # 获取全局回复风格 (长期记忆)
POST /api/memory/style              # 手动覆盖全局回复风格
DELETE /api/memory/style            # 清除当前全局回复风格
```

### 2.3 认证中间件 — `whaleclaw/gateway/middleware.py`

```python
class AuthMiddleware:
    """
    认证模式:
    1. token: 请求头 Authorization: Bearer <token>
    2. password: POST /api/auth/login 获取 JWT token
    3. none: 无认证 (仅限 localhost)
    """
```

配置:
```json
{
  "gateway": {
    "auth": {
      "mode": "password",
      "password": "your-secret"
    }
  }
}
```

当 `bind` 为 `127.0.0.1` 且未配置认证时，默认无认证。
当 `bind` 为 `0.0.0.0` 时，必须配置认证，否则拒绝启动。

---

## 3. WebChat 前端

### 3.1 技术选型

- 框架: Vue 3 + Composition API
- 构建: Vite
- UI: 自定义 CSS (不引入重型 UI 库)
- Markdown: markdown-it + highlight.js
- WebSocket: 原生 WebSocket API
- 构建产物: 打包为静态文件，由 FastAPI 的 `StaticFiles` 提供

前端代码位置: `whaleclaw/web/frontend/`
构建产物: `whaleclaw/web/static/`

### 3.2 页面结构

```
┌─────────────────────────────────────────────┐
│  WhaleClaw                    [新建会话] [⚙] │
├──────────┬──────────────────────────────────┤
│          │                                  │
│ 会话列表  │        聊天区域                   │
│          │                                  │
│ ○ 会话1  │  ┌─────────────────────────┐     │
│ ● 会话2  │  │ 用户: 你好               │     │
│ ○ 会话3  │  │                         │     │
│          │  │ Agent: 你好！我是...     │     │
│          │  │ ┌─ 工具调用 ──────────┐  │     │
│          │  │ │ bash: ls -la       │  │     │
│          │  │ │ > file1.py         │  │     │
│          │  │ │ > file2.py         │  │     │
│          │  │ └────────────────────┘  │     │
│          │  └─────────────────────────┘     │
│          │                                  │
│          │  ┌──────────────────────┐ [发送] │
│          │  │ 输入消息...           │        │
│          │  └──────────────────────┘        │
└──────────┴──────────────────────────────────┘
```

### 3.3 核心组件

#### ChatView.vue — 主聊天视图
- 消息列表 (自动滚动到底部)
- 输入框 (支持 Shift+Enter 换行, Enter 发送)
- 流式消息渲染 (打字机效果)

#### MessageBubble.vue — 消息气泡
- 用户消息: 右对齐，蓝色背景
- Agent 消息: 左对齐，灰色背景
- Markdown 渲染 (支持表格/列表/链接)
- 代码块: 语法高亮 + 复制按钮

#### ToolCallCard.vue — 工具调用卡片
- 折叠/展开
- 显示工具名称、参数、执行结果
- 执行中显示 loading 动画

#### SessionList.vue — 会话列表侧边栏
- 会话列表 (按最近更新排序)
- 新建/删除会话
- 当前会话高亮

#### SettingsPanel.vue — 设置面板
- 模型选择下拉框
- 思考深度滑块
- 主题切换 (亮/暗)
- 全局回复风格设置 (自动生成，可手动覆盖/清除)

### 3.4 WebSocket 客户端

```typescript
class WhaleclawWS {
  connect(sessionId: string): void
  send(content: string): void
  onStream(callback: (chunk: string) => void): void
  onToolCall(callback: (data: ToolCallData) => void): void
  onToolResult(callback: (data: ToolResultData) => void): void
  onMessage(callback: (data: MessageData) => void): void
  onError(callback: (error: ErrorData) => void): void
  disconnect(): void
}
```

自动重连: 断线后指数退避重连 (1s, 2s, 4s, 8s, 最大 30s)。

### 3.5 主题

支持亮色/暗色主题，默认跟随系统:

- 亮色: 白色背景，深色文字
- 暗色: 深灰背景 (#1a1a2e)，浅色文字
- 强调色: #4a9eff (蓝色)

---

## 4. 静态文件服务

### 4.1 Gateway 集成

```python
# whaleclaw/gateway/app.py
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="whaleclaw/web/static"), name="static")

@app.get("/")
async def index():
    """返回 WebChat SPA 入口 HTML"""
    return FileResponse("whaleclaw/web/static/index.html")
```

### 4.2 构建流程

```bash
cd whaleclaw/web/frontend
npm install
npm run build    # 输出到 ../static/
```

构建产物提交到仓库 (或在安装时自动构建)。

---

## 5. 媒体支持

### 5.1 文件上传

```
POST /api/upload
Content-Type: multipart/form-data

返回: {"url": "/api/files/{file_id}", "filename": "...", "size": ...}
```

上传的文件存储在 `~/.whaleclaw/uploads/`。

### 5.2 图片预览

消息中的图片 URL 自动渲染为可点击的缩略图。
支持粘贴图片直接上传。

---

## 验收标准

### AC-1: 浏览器访问
打开 `http://127.0.0.1:18666/`，显示 WebChat 界面。

### AC-2: 发送消息并收到流式回复
在输入框输入消息，点击发送:
- 消息出现在右侧气泡
- Agent 回复逐字出现在左侧气泡 (流式)
- Markdown 正确渲染 (代码块有高亮)

### AC-3: 工具调用可视化
发送 "列出当前目录的文件":
- 显示工具调用卡片 (bash: ls)
- 卡片内显示执行结果
- 最终 Agent 回复引用工具结果

### AC-4: 会话管理
- 点击 "新建会话" 创建空会话
- 在侧边栏切换会话，消息历史正确加载
- 删除会话后从列表消失

### AC-5: 密码认证
配置 `gateway.auth.mode = "password"`:
- 访问页面显示登录表单
- 输入正确密码后进入聊天界面
- 错误密码显示错误提示

### AC-6: 暗色主题
- 切换到暗色主题，所有元素正确显示
- 代码块在暗色主题下可读

### AC-7: 移动端适配
- 手机浏览器访问，布局自适应
- 侧边栏可折叠

### AC-8: 全局回复风格设置
- 设置面板可查看当前长期记忆提炼出的全局回复风格
- 支持手动覆盖并保存
- 支持清除当前全局风格

---

## 文件清单

```
whaleclaw/channels/__init__.py          (更新)
whaleclaw/channels/base.py
whaleclaw/channels/manager.py
whaleclaw/channels/webchat/__init__.py
whaleclaw/channels/webchat/handler.py
whaleclaw/gateway/middleware.py          (更新: 认证)
whaleclaw/gateway/app.py                (更新: REST API + 静态文件)
whaleclaw/web/frontend/                 (Vue 3 项目)
  package.json
  vite.config.ts
  index.html
  src/
    main.ts
    App.vue
    components/
      ChatView.vue
      MessageBubble.vue
      ToolCallCard.vue
      SessionList.vue
      SettingsPanel.vue
      LoginForm.vue
    composables/
      useWebSocket.ts
      useSession.ts
      useAuth.ts
      useTheme.ts
    styles/
      main.css
      themes.css
    types.ts
whaleclaw/web/static/                   (构建产物)
tests/test_channels/test_webchat.py
tests/test_channels/test_base.py
tests/test_gateway/test_middleware.py
```
