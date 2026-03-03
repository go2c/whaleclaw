"""PPT edit tool — edit text/style/image/notes in an existing .pptx file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.tools.deps import ensure_tool_dep


class PptEditTool(Tool):
    """Edit an existing PPT by actions on a specific slide."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ppt_edit",
            description=(
                "修改现有 PPT（.pptx）：支持文本替换、标题更新、商务风格、背景色、备注、插图。"
            ),
            parameters=[
                ToolParameter(name="path", type="string", description="PPT 文件绝对路径"),
                ToolParameter(name="slide_index", type="integer", description="页码（从 1 开始）"),
                ToolParameter(
                    name="action",
                    type="string",
                    description=(
                        "操作类型：replace_text|set_title|set_notes|set_background|add_image|apply_business_style"
                    ),
                    required=False,
                    enum=[
                        "replace_text",
                        "set_title",
                        "set_notes",
                        "set_background",
                        "add_image",
                        "apply_business_style",
                    ],
                ),
                ToolParameter(
                    name="old_text",
                    type="string",
                    description="旧文案（replace_text 时必填）",
                    required=False,
                ),
                ToolParameter(
                    name="new_text",
                    type="string",
                    description="新文案（replace_text/set_title/set_notes 时使用）",
                    required=False,
                ),
                ToolParameter(
                    name="image_path",
                    type="string",
                    description="插图绝对路径（add_image 时必填）",
                    required=False,
                ),
                ToolParameter(
                    name="image_left",
                    type="number",
                    description="插图左边距（英寸，默认 1.0）",
                    required=False,
                ),
                ToolParameter(
                    name="image_top",
                    type="number",
                    description="插图上边距（英寸，默认 1.8）",
                    required=False,
                ),
                ToolParameter(
                    name="image_width",
                    type="number",
                    description="插图宽度（英寸，默认 6.5）",
                    required=False,
                ),
                ToolParameter(
                    name="background_color",
                    type="string",
                    description="背景色（十六进制，如 #0F2747）",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not ensure_tool_dep("pptx"):
            return ToolResult(success=False, output="", error="缺少依赖 python-pptx")

        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE_TYPE
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Inches, Pt

        raw_path = str(kwargs.get("path", "")).strip()
        action = str(kwargs.get("action", "replace_text")).strip().lower() or "replace_text"
        old_text = str(kwargs.get("old_text", "")).strip()
        new_text = str(kwargs.get("new_text", ""))
        image_path = str(kwargs.get("image_path", "")).strip()
        background_color = str(kwargs.get("background_color", "")).strip()
        try:
            slide_index = int(kwargs.get("slide_index", 0))
        except (TypeError, ValueError):
            slide_index = 0

        if not raw_path:
            return ToolResult(success=False, output="", error="path 不能为空")
        if slide_index <= 0:
            return ToolResult(success=False, output="", error="slide_index 必须 >= 1")

        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            return ToolResult(success=False, output="", error=f"文件不存在: {path}")
        if path.suffix.lower() != ".pptx":
            return ToolResult(success=False, output="", error="仅支持 .pptx 文件")

        try:
            prs = Presentation(str(path))
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"PPT 打开失败: {exc}")

        if slide_index > len(prs.slides):
            return ToolResult(
                success=False,
                output="",
                error=f"页码越界: slide_index={slide_index}, 总页数={len(prs.slides)}",
            )

        slide = prs.slides[slide_index - 1]

        if action == "replace_text":
            if not old_text:
                return ToolResult(success=False, output="", error="old_text 不能为空")
            replaced = 0
            for shape in slide.shapes:
                if not hasattr(shape, "text"):
                    continue
                text = shape.text or ""
                count = text.count(old_text)
                if count <= 0:
                    continue
                shape.text = text.replace(old_text, new_text)
                replaced += count
            if replaced == 0:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"第 {slide_index} 页未找到匹配文本",
                )
            output = f"已修改 {path} 第 {slide_index} 页，替换 {replaced} 处文本"
        elif action == "set_title":
            title_shape = slide.shapes.title
            if title_shape is None:
                title_shape = slide.shapes.add_textbox(
                    Inches(0.8),
                    Inches(0.35),
                    Inches(11.0),
                    Inches(1.0),
                )
            tf = title_shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = new_text
            p.alignment = PP_ALIGN.LEFT
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(34)
                run.font.name = "Microsoft YaHei"
                run.font.color.rgb = RGBColor(15, 39, 71)
            output = f"已更新 {path} 第 {slide_index} 页标题"
        elif action == "set_notes":
            notes = slide.notes_slide.notes_text_frame
            notes.clear()
            notes.text = new_text
            output = f"已更新 {path} 第 {slide_index} 页备注"
        elif action == "set_background":
            hex_color = (background_color or "#F4F7FB").strip().lstrip("#")
            if len(hex_color) != 6:
                return ToolResult(
                    success=False,
                    output="",
                    error="background_color 必须是 6 位十六进制",
                )
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            except ValueError:
                return ToolResult(
                    success=False,
                    output="",
                    error="background_color 不是合法十六进制颜色",
                )
            fill = slide.background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(r, g, b)
            output = f"已设置 {path} 第 {slide_index} 页背景色 #{hex_color.upper()}"
        elif action == "add_image":
            if not image_path:
                return ToolResult(success=False, output="", error="add_image 需要 image_path")
            img = Path(image_path).expanduser().resolve()
            if not img.is_file():
                return ToolResult(success=False, output="", error=f"图片不存在: {img}")
            left = Inches(float(kwargs.get("image_left", 1.0)))
            top = Inches(float(kwargs.get("image_top", 1.8)))
            width = Inches(float(kwargs.get("image_width", 6.5)))
            slide.shapes.add_picture(str(img), left, top, width=width)
            output = f"已在 {path} 第 {slide_index} 页插入图片 {img.name}"
        elif action == "apply_business_style":
            def _iter_shapes_recursive(container: Any) -> list[Any]:
                items: list[Any] = []
                for shp in container:
                    items.append(shp)
                    if getattr(shp, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
                        inner = getattr(shp, "shapes", None)
                        if inner is not None:
                            items.extend(_iter_shapes_recursive(inner))
                return items

            fill = slide.background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(245, 248, 252)
            dark_bar_restyled = 0
            for shape in _iter_shapes_recursive(slide.shapes):
                if not hasattr(shape, "fill"):
                    continue
                shape_fill = shape.fill
                if getattr(shape_fill, "type", None) != 1:  # MSO_FILL.SOLID
                    continue
                color = getattr(shape_fill, "fore_color", None)
                rgb = getattr(color, "rgb", None)
                if rgb is None:
                    continue
                # Treat large dark rectangles/bars as candidate overlays.
                is_dark = rgb[0] <= 50 and rgb[1] <= 50 and rgb[2] <= 50
                is_large_bar = (
                    getattr(shape, "width", 0) >= int(prs.slide_width * 0.35)
                    and getattr(shape, "height", 0) <= int(prs.slide_height * 0.75)
                )
                if not (is_dark and is_large_bar):
                    continue
                shape_fill.solid()
                shape_fill.fore_color.rgb = RGBColor(214, 228, 245)
                if hasattr(shape_fill, "transparency"):
                    shape_fill.transparency = 0.15
                if hasattr(shape, "line") and getattr(shape.line, "fill", None) is not None:
                    shape.line.fill.background()
                dark_bar_restyled += 1
            if slide.shapes.title is not None:
                ttf = slide.shapes.title.text_frame
                if ttf.paragraphs:
                    p = ttf.paragraphs[0]
                    for run in p.runs:
                        run.font.bold = True
                        run.font.size = Pt(34)
                        run.font.name = "Microsoft YaHei"
                        run.font.color.rgb = RGBColor(15, 39, 71)
            output = (
                f"已应用 {path} 第 {slide_index} 页商务风格，"
                f"重设深色条 {dark_bar_restyled} 处"
            )
        else:
            return ToolResult(success=False, output="", error=f"不支持的 action: {action}")

        try:
            prs.save(str(path))
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"PPT 保存失败: {exc}")

        return ToolResult(success=True, output=output)
