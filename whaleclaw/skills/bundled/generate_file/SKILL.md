---
triggers:
  - PPT
  - pptx
  - Excel
  - xlsx
  - PDF
  - 文档
  - 表格
  - 幻灯片
  - 演示文稿
  - 报告
  - 简历
  - resume
  - spreadsheet
  - 做个
  - 生成
max_tokens: 1200
---

# 生成文件（PPT/Excel/PDF/Word）

## 效率要求（最重要）

**流程：**
1. **先回复用户**：简要说明你打算做什么（几页、什么内容、预计需要的时间），让用户知道你在处理
2. `browser` → 搜索并下载与主题相关的图片，数量自行判断
3. `file_write` → 写完整 Python 脚本到 `/tmp/gen_xxx.py`（脚本中用 `add_picture()` 插入已下载的图片）
4. `bash` → 执行脚本 `./python/bin/python3.12 /tmp/gen_xxx.py`
5. 告诉用户文件路径

**严禁：**
- 不要用 `python -c '...'` 或 `python3 -c '...'` — 脚本太长会截断
- 不要写完脚本后又 file_edit 修改 — 一次写对
- 不要执行失败后反复修改重试 — 检查好再执行
- 不要分多次 file_write — 一次写完整个脚本

## PPT (python-pptx) 规范

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
```

每页必须有具体内容，不是只有标题。

配色推荐：主色 `RGBColor(0x2B, 0x57, 0x9A)`，辅色 `RGBColor(0x5B, 0x9B, 0xD5)`。

页面类型：
1. **封面页**: 大标题 + 副标题 + 色条装饰
2. **内容页**: 顶部色条标题 + 正文要点（每个要点有 1-3 句具体描述）
3. **表格页**（可选）: `slide.shapes.add_table()` 填入真实数据
4. **总结页**: 要点回顾

字号：标题 28-36pt 加粗，正文 16-20pt。

内容要求：
```
❌ 错误: "第一天：鼓浪屿风情"
✅ 正确: "第一天：鼓浪屿风情
   上午: 渡轮前往（35元/人），游日光岩（60元）→ 菽庄花园（30元）
   午餐: 龙头路小吃街（海蛎煎、沙茶面）
   预算: 约 200-300 元/人"
```

**图片处理（重要）：**
- 先用 `browser` 工具搜索并下载 1~3 张与主题相关的图片
- 下载后在脚本中用 `prs.slides[i].shapes.add_picture(图片路径, ...)` 插入
- 图片路径必须使用 browser 工具返回的真实路径，严禁编造
- 如果用户也上传了图片，优先使用用户的图片
- 图片建议放在封面页或内容页作为配图，尺寸适当缩放
**封面底图遮挡（必须遵守）：**
- 禁止用全屏纯色矩形覆盖底图（尤其是黑色块）
- 背景色只能用 `slide.background.fill` 设置
- 若需要提升文字可读性，只允许“小面积半透明条/阴影”，且必须在图片之后添加并保持可见底图

## Excel / PDF / Word

- Excel: 首行冻结+加粗+背景色，列宽自适应，数据有边框
- PDF: 注册中文字体（PingFang.ttc），合理边距
- Word: Heading 样式分层级，段落有缩进

## 关键提醒

- 脚本必须完整可运行，写之前在脑中过一遍有没有语法错误
- 不需要安装依赖（已预装 python-pptx / openpyxl / reportlab / python-docx）
- 保存到 `/tmp/<有意义的文件名>.<后缀>`
- 文件名必须去除空格与特殊符号，仅保留中文/英文/数字/下划线，例如：`/tmp/同安2日游.pptx`
- 用户说"做个 PPT"= 完整可用的文件，不是骨架
