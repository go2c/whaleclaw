# Phase 8: 桌面/移动端 + 运维

## 目标

完善 WhaleClaw 的终端用户体验和生产运维能力:
1. 桌面应用 — Tauri 包装 WebChat + 系统托盘 + 全局快捷键
2. 移动端 — Flutter 客户端 (Canvas + 语音)
3. 守护进程 — systemd/launchd 服务管理
4. 健康检查 — `whaleclaw doctor` 全面诊断
5. 远程访问 — Tailscale/SSH 隧道
6. Canvas — Agent 驱动的可视化工作区
7. 节点系统 — 设备能力注册和远程调用

## 前置条件

- Phase 1 ~ 7 全部完成

---

## 1. 桌面应用 (Tauri)

### 1.1 技术选型

- 框架: Tauri v2 (Rust 后端 + Web 前端)
- 前端: 复用 Phase 3 的 WebChat Vue 3 SPA
- 优势: 体积小 (~10MB)、原生性能、系统 API 访问

### 1.2 功能

#### 系统托盘
```
[WhaleClaw 图标]
├── 打开 WhaleClaw
├── ─────────────
├── Gateway 状态: 运行中
├── 启动 Gateway
├── 停止 Gateway
├── ─────────────
├── 设置
└── 退出
```

#### 全局快捷键
- `Cmd+Shift+W` (macOS) / `Ctrl+Shift+W` (Windows/Linux): 打开/隐藏窗口
- `Cmd+Shift+N`: 新建会话
- 可自定义

#### 窗口行为
- 关闭窗口 = 最小化到托盘 (不退出)
- 支持 Always on Top (置顶)
- 记忆窗口位置和大小

#### 原生通知
- Agent 回复时推送系统通知 (窗口不在前台时)
- 定时提醒通过系统通知推送

### 1.3 项目结构

```
apps/desktop/
  src-tauri/
    src/
      main.rs                    # Tauri 入口
      tray.rs                    # 系统托盘
      commands.rs                # Tauri 命令 (启动/停止 Gateway)
      shortcuts.rs               # 全局快捷键
    tauri.conf.json
    Cargo.toml
  src/                           # 前端 (symlink 到 whaleclaw/web/frontend/src)
  package.json
```

### 1.4 Gateway 管理

桌面应用内嵌 Gateway 管理:
- 启动时自动启动 Gateway (如果未运行)
- 托盘菜单控制 Gateway 启停
- 显示 Gateway 日志 (调试面板)

---

## 2. 移动端 (Flutter)

### 2.1 技术选型

- 框架: Flutter 3.x
- 状态管理: Riverpod
- 平台: iOS + Android

### 2.2 功能

#### 核心
- 聊天界面 (与 WebChat 功能对齐)
- WebSocket 连接到 Gateway
- 推送通知 (FCM / APNs)

#### Canvas
- Agent 驱动的可视化工作区
- 支持 HTML/CSS/JS 渲染
- 交互式组件

#### 语音
- 语音输入 (STT)
- 语音输出 (TTS)
- 连续对话模式 (Talk Mode)

#### 设备能力
- 摄像头 (拍照/录像)
- 屏幕录制
- 位置信息
- 通知

### 2.3 项目结构

```
apps/mobile/
  lib/
    main.dart
    screens/
      chat_screen.dart
      canvas_screen.dart
      settings_screen.dart
    widgets/
      message_bubble.dart
      tool_call_card.dart
    services/
      websocket_service.dart
      notification_service.dart
      voice_service.dart
      camera_service.dart
    models/
      message.dart
      session.dart
    providers/
      chat_provider.dart
      settings_provider.dart
  pubspec.yaml
```

### 2.4 设备节点注册

移动端作为 "节点" 注册到 Gateway:

```python
# Gateway 端
class DeviceNode(BaseModel):
    id: str
    name: str
    platform: str                    # "ios", "android", "macos"
    capabilities: list[str]          # ["camera", "screen_record", "location", "notification"]
    connected_at: datetime
    last_heartbeat: datetime
```

---

## 3. 守护进程

### 3.1 macOS (launchd)

```python
class LaunchdService:
    """macOS launchd 服务管理"""

    PLIST_PATH = Path.home() / "Library/LaunchAgents/ai.whaleclaw.gateway.plist"

    def install(self, config: GatewayConfig) -> None:
        """生成 plist 并安装到 LaunchAgents"""

    def uninstall(self) -> None:
        """移除 plist"""

    def start(self) -> None:
        """launchctl load"""

    def stop(self) -> None:
        """launchctl unload"""

    def status(self) -> ServiceStatus: ...
```

plist 模板:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ...>
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.whaleclaw.gateway</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python3.12</string>
        <string>-m</string>
        <string>whaleclaw</string>
        <string>gateway</string>
        <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>~/.whaleclaw/logs/gateway.log</string>
    <key>StandardErrorPath</key>
    <string>~/.whaleclaw/logs/gateway.err</string>
</dict>
</plist>
```

### 3.2 Linux (systemd)

```python
class SystemdService:
    """Linux systemd 用户服务管理"""

    UNIT_PATH = Path.home() / ".config/systemd/user/whaleclaw-gateway.service"

    def install(self, config: GatewayConfig) -> None: ...
    def uninstall(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> ServiceStatus: ...
```

### 3.3 CLI 集成

```bash
whaleclaw gateway --install-daemon     # 安装并启动守护进程
whaleclaw gateway --uninstall-daemon   # 卸载守护进程
whaleclaw gateway status               # 查看守护进程状态
```

---

## 4. 健康检查

### 4.1 Doctor 命令 — `whaleclaw/cli/doctor_cmd.py`

```bash
whaleclaw doctor
```

检查项:

```
WhaleClaw Doctor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[检查项]                        [状态]
Python 版本                     ✅ 3.12.4
配置文件                        ✅ ~/.whaleclaw/whaleclaw.json
Gateway 端口                    ✅ 18666 可用
Anthropic API Key               ✅ 已配置
OpenAI API Key                  ⚠️ 未配置
DeepSeek API Key                ✅ 已配置
通义千问 API Key                ❌ 未配置
SQLite 数据库                   ✅ 正常
飞书应用                        ✅ 已配置 (App ID: cli_xxx)
飞书 Webhook                    ✅ 可达
WebChat 前端                    ✅ 已构建
Docker                          ⚠️ 未安装 (沙箱不可用)
ChromaDB                        ✅ 正常
守护进程                        ✅ 运行中
EvoMap 节点                     ✅ 已注册 (reputation: 42)
EvoMap 同步                     ✅ 最近同步: 2小时前
磁盘空间                        ✅ 充足 (12.3 GB 可用)
日志大小                        ⚠️ 日志较大 (234 MB)，建议清理

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 8 通过  ⚠️ 3 警告  ❌ 1 错误
```

### 4.2 检查器框架

```python
class HealthCheck(ABC):
    name: str
    @abstractmethod
    async def check(self) -> CheckResult: ...

class CheckResult(BaseModel):
    status: Literal["ok", "warning", "error"]
    message: str
    details: str | None = None
    fix_hint: str | None = None      # 修复建议

class Doctor:
    checks: list[HealthCheck]
    async def run_all(self) -> list[CheckResult]: ...
```

---

## 5. 远程访问

### 5.1 Tailscale 集成

```python
class TailscaleManager:
    """Tailscale Serve/Funnel 管理"""

    async def setup_serve(self, port: int) -> str:
        """配置 Tailscale Serve (tailnet 内访问)"""

    async def setup_funnel(self, port: int) -> str:
        """配置 Tailscale Funnel (公网访问)"""

    async def teardown(self) -> None:
        """移除 Serve/Funnel 配置"""
```

配置:
```json
{
  "gateway": {
    "tailscale": {
      "mode": "serve",              // off / serve / funnel
      "reset_on_exit": true
    }
  }
}
```

### 5.2 SSH 隧道

```python
class SSHTunnel:
    """SSH 反向隧道"""

    async def start(self, remote_host: str, remote_port: int, local_port: int) -> None: ...
    async def stop(self) -> None: ...
```

---

## 6. Canvas

### 6.1 Canvas 概念

Agent 驱动的可视化工作区:
- Agent 可以推送 HTML/CSS/JS 到 Canvas
- 用户可以与 Canvas 交互
- 适用于: 数据可视化、表单、游戏、演示文稿

### 6.2 Canvas 工具 — `whaleclaw/tools/canvas.py`

```python
class CanvasTool(Tool):
    """Canvas 可视化工作区工具"""

    # 子命令:
    # - push: 推送 HTML 内容到 Canvas
    # - reset: 重置 Canvas
    # - snapshot: 截取 Canvas 截图
    # - eval: 在 Canvas 中执行 JavaScript
```

### 6.3 Canvas 协议

WebSocket 新增消息类型:

```python
class MessageType(str, Enum):
    # ... 已有类型
    CANVAS_PUSH = "canvas_push"      # 推送 Canvas 内容
    CANVAS_RESET = "canvas_reset"    # 重置 Canvas
    CANVAS_EVENT = "canvas_event"    # Canvas 交互事件
```

---

## 7. 节点系统

### 7.1 节点注册

设备 (macOS/iOS/Android) 作为节点注册到 Gateway:

```python
class NodeManager:
    """设备节点管理"""

    async def register(self, node: DeviceNode) -> None: ...
    async def unregister(self, node_id: str) -> None: ...
    async def list_nodes(self) -> list[DeviceNode]: ...
    async def invoke(self, node_id: str, action: str, params: dict) -> Any: ...
```

### 7.2 节点能力

```python
class NodeCapability(str, Enum):
    CAMERA_SNAP = "camera.snap"
    CAMERA_CLIP = "camera.clip"
    SCREEN_RECORD = "screen.record"
    LOCATION_GET = "location.get"
    NOTIFICATION = "notification"
    SYSTEM_RUN = "system.run"
```

### 7.3 节点工具

```python
class NodeInvokeTool(Tool):
    """调用设备节点能力"""
    # 参数: node_id, action, params
    # 示例: 调用 iPhone 拍照 -> node.invoke("iphone-1", "camera.snap", {})
```

---

## 8. 引导设置向导

### 8.1 Onboard 命令

```bash
whaleclaw onboard
```

交互式向导:
1. 欢迎页 + 项目介绍
2. 检查 Python 环境
3. 配置 AI 模型 (选择 Provider + 输入 API Key)
4. 配置消息渠道 (WebChat / 飞书)
5. 安全设置 (认证方式 / DM 策略)
6. EvoMap 设置 (可选 — 注册节点 + 显示 claim code 绑定账户)
7. 安装守护进程 (可选)
8. 启动 Gateway + 打开 WebChat
9. 发送测试消息

### 8.2 向导 UI

使用 Rich + Typer 的交互式 TUI:
- 进度条显示当前步骤
- 彩色输出
- 输入验证
- 错误恢复

---

## 验收标准

### AC-1: 桌面应用
- macOS 上双击打开 WhaleClaw.app
- 系统托盘显示图标和菜单
- 全局快捷键唤起窗口
- Gateway 自动启动

### AC-2: 移动端
- iOS/Android 上安装并打开 WhaleClaw
- 能连接到 Gateway 并聊天
- 语音输入/输出正常
- 推送通知正常

### AC-3: 守护进程
```bash
whaleclaw gateway --install-daemon
# Gateway 作为系统服务运行
# 重启电脑后自动启动
```

### AC-4: Doctor 诊断
```bash
whaleclaw doctor
# 输出所有检查项的状态
# 对错误项给出修复建议
```

### AC-5: 远程访问
配置 Tailscale Serve 后:
- 同一 tailnet 内的设备可访问 WebChat
- 认证正常工作

### AC-6: Canvas
```
用户: 画一个柱状图显示销售数据
Agent: [调用 canvas 工具: push HTML]
# WebChat/桌面应用中显示交互式图表
```

### AC-7: 引导向导
```bash
whaleclaw onboard
# 交互式完成所有配置
# 最终成功发送测试消息
```

---

## 文件清单

```
# 桌面应用
apps/desktop/src-tauri/src/main.rs
apps/desktop/src-tauri/src/tray.rs
apps/desktop/src-tauri/src/commands.rs
apps/desktop/src-tauri/src/shortcuts.rs
apps/desktop/src-tauri/tauri.conf.json
apps/desktop/src-tauri/Cargo.toml
apps/desktop/package.json

# 移动端
apps/mobile/lib/main.dart
apps/mobile/lib/screens/chat_screen.dart
apps/mobile/lib/screens/canvas_screen.dart
apps/mobile/lib/services/websocket_service.dart
apps/mobile/lib/services/voice_service.dart
apps/mobile/pubspec.yaml

# 守护进程
whaleclaw/daemon/__init__.py
whaleclaw/daemon/launchd.py
whaleclaw/daemon/systemd.py
whaleclaw/daemon/manager.py

# 健康检查
whaleclaw/cli/doctor_cmd.py
whaleclaw/doctor/__init__.py
whaleclaw/doctor/checks.py
whaleclaw/doctor/runner.py

# 远程访问
whaleclaw/remote/__init__.py
whaleclaw/remote/tailscale.py
whaleclaw/remote/ssh_tunnel.py

# Canvas
whaleclaw/tools/canvas.py
whaleclaw/canvas/__init__.py
whaleclaw/canvas/host.py

# 节点
whaleclaw/nodes/__init__.py
whaleclaw/nodes/manager.py
whaleclaw/nodes/protocol.py
whaleclaw/tools/node_invoke.py

# 引导向导
whaleclaw/wizard/__init__.py
whaleclaw/wizard/onboard.py
whaleclaw/wizard/steps.py
whaleclaw/cli/onboard_cmd.py

# 测试
tests/test_daemon/test_launchd.py
tests/test_daemon/test_systemd.py
tests/test_doctor/test_checks.py
tests/test_remote/test_tailscale.py
tests/test_canvas/test_host.py
tests/test_nodes/test_manager.py
tests/test_wizard/test_onboard.py
```
