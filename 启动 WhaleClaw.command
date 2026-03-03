#!/bin/bash
# ═══════════════════════════════════════════════
#  WhaleClaw — 启动 Gateway
# ═══════════════════════════════════════════════
cd "$(dirname "$0")"

PYTHON="./python/bin/python3.12"
PROJECT_META_FILE=".whaleclaw.project.json"
DEFAULT_WHALECLAW_HOME="$HOME/.whaleclaw"
WHALECLAW_HOME_DIR="$DEFAULT_WHALECLAW_HOME"

# 自动检测本地代理（用于访问海外 API）
if [ -z "$https_proxy" ]; then
    for port in 7897 7890 1087 8080; do
        if nc -z 127.0.0.1 $port 2>/dev/null; then
            export https_proxy="http://127.0.0.1:$port"
            export http_proxy="http://127.0.0.1:$port"
            break
        fi
    done
fi

if [ ! -f "$PYTHON" ]; then
    echo ""
    echo "  ❌ 未找到内嵌 Python: $PYTHON"
    echo "  请确认项目目录完整。"
    read -p "  按回车键退出..."
    exit 1
fi

# 读取当前项目绑定的数据根目录（若存在）
if [ -f "$PROJECT_META_FILE" ]; then
    bound_home=$("$PYTHON" -c "
import json, pathlib
p = pathlib.Path('$PROJECT_META_FILE')
try:
    data = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    data = {}
home = data.get('whaleclaw_home') if isinstance(data, dict) else ''
print(home or '')
" 2>/dev/null)
    if [ -n "$bound_home" ]; then
        WHALECLAW_HOME_DIR="$bound_home"
    fi
fi
export WHALECLAW_HOME="$WHALECLAW_HOME_DIR"

# 安装依赖（首次运行）
if ! "$PYTHON" -c "import whaleclaw" 2>/dev/null; then
    echo ""
    echo "  📦 首次运行，正在安装依赖..."
    "$PYTHON" -m pip install -e ".[dev]" --quiet
    echo "  ✅ 依赖安装完成"
fi

# 读取端口和绑定地址（从配置文件）
eval $("$PYTHON" -c "
import json, pathlib, os
home = os.environ.get('WHALECLAW_HOME', str(pathlib.Path.home() / '.whaleclaw'))
p = pathlib.Path(home).expanduser() / 'whaleclaw.json'
port, bind = 18666, '127.0.0.1'
if p.exists():
    cfg = json.loads(p.read_text())
    gw = cfg.get('gateway', {})
    port = gw.get('port', port)
    bind = gw.get('bind', bind)
print(f'PORT={port}')
print(f'BIND={bind}')
" 2>/dev/null)

# 释放被占用的端口
OLD_PID=$(lsof -ti :${PORT} 2>/dev/null)
if [ -n "$OLD_PID" ]; then
    echo ""
    echo "  ⚠️  端口 ${PORT} 被占用 (PID: ${OLD_PID})，正在释放..."
    kill -9 $OLD_PID 2>/dev/null
    sleep 1
    echo "  ✅ 端口已释放"
fi

echo ""
echo "  🐋 WhaleClaw Gateway 正在启动..."
echo "  🎉 B站飞翔鲸祝您马年大吉！财源广进！WhaleClaw 免费开源！"
echo "  ─────────────────────────────────"
echo ""
echo "  🌐 WebChat:  http://${BIND}:${PORT}"
echo "  📡 API:      http://${BIND}:${PORT}/api/status"
echo "  🔌 WS:       ws://${BIND}:${PORT}/ws"
echo "  🗂️  根目录:   ${WHALECLAW_HOME}"
echo ""
echo "  按 Ctrl+C 停止服务"
echo "  ─────────────────────────────────"
echo ""

# 等待服务真正就绪后再自动打开浏览器（避免刚启动时出现 connection refused）
(
    for _ in $(seq 1 120); do
        # 注意：/api/status 可能返回 401/403，-f 会导致 curl 失败
        if curl -sS "http://${BIND}:${PORT}/api/status" >/dev/null 2>&1; then
            sleep 2
            open "http://${BIND}:${PORT}"
            exit 0
        fi
        sleep 0.5
    done
) &

exec "$PYTHON" -m whaleclaw gateway run
