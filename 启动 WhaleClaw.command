#!/bin/bash
# ═══════════════════════════════════════════════
#  WhaleClaw — 启动 Gateway
# ═══════════════════════════════════════════════
cd "$(dirname "$0")"

PYTHON="./python/bin/python3.12"

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

# 安装依赖（首次运行）
if ! "$PYTHON" -c "import whaleclaw" 2>/dev/null; then
    echo ""
    echo "  📦 首次运行，正在安装依赖..."
    "$PYTHON" -m pip install -e ".[dev]" --quiet
    echo "  ✅ 依赖安装完成"
fi

# 读取端口和绑定地址（从配置文件）
eval $("$PYTHON" -c "
import json, pathlib
p = pathlib.Path.home() / '.whaleclaw/whaleclaw.json'
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
echo "  ─────────────────────────────────"
echo ""
echo "  🌐 WebChat:  http://${BIND}:${PORT}"
echo "  📡 API:      http://${BIND}:${PORT}/api/status"
echo "  🔌 WS:       ws://${BIND}:${PORT}/ws"
echo ""
echo "  按 Ctrl+C 停止服务"
echo "  ─────────────────────────────────"
echo ""

# 2 秒后自动打开浏览器
(sleep 2 && open "http://${BIND}:${PORT}") &

exec "$PYTHON" -m whaleclaw gateway run
