"""Multi-agent orchestration for WhaleClaw."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, cast

from whaleclaw.agent import single_agent as single
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.router import ModelRouter
from whaleclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from whaleclaw.agent.single_agent import ImageContent, OnRoundResult, StreamCallback

run_agent = single.run_agent
MULTI_AGENT_SCENARIO_LABELS = single.MULTI_AGENT_SCENARIO_LABELS
scenario_discuss_focus = single.scenario_discuss_focus
truncate_to_tokens = single.truncate_to_tokens
select_native_tool_names = single.select_native_tool_names
extract_round_delivery_section = single.extract_round_delivery_section
ABS_FILE_PATH_RE = single.ABS_FILE_PATH_RE
NON_DELIVERY_EXTS = single.NON_DELIVERY_EXTS
OFFICE_PATH_RE = single.OFFICE_PATH_RE
COORDINATOR_ASK_RE = single.COORDINATOR_ASK_RE
extract_delivery_artifact_paths = single.extract_delivery_artifact_paths
fix_version_suffix = single.fix_version_suffix
snapshot_round_artifacts = single.snapshot_round_artifacts
extract_artifact_baseline = single.extract_artifact_baseline


async def run_multi_agent_controller_discussion(
    *,
    user_message: str,
    pending_topic: str,
    cfg: dict[str, object],
    session_id: str,
    config: WhaleclawConfig,
    router: ModelRouter,
    registry: ToolRegistry,
    extra_memory: str,
    trigger_event_id: str,
    trigger_text_preview: str,
    include_intro: bool,
) -> str:
    roles = [
        role
        for role in cast(list[dict[str, object]], cfg["roles"])
        if bool(role.get("enabled", True))
    ]
    mode = cast(str, cfg["mode"])
    rounds = cast(int, cfg["max_rounds"])
    scenario = str(cfg.get("scenario", "software_development")).strip()
    scenario_label = MULTI_AGENT_SCENARIO_LABELS.get(scenario, scenario or "自定义")
    discuss_focus = scenario_discuss_focus(scenario)
    role_lines: list[str] = []
    role_bullets: list[str] = []
    for role in roles:
        name = str(role.get("name", role.get("id", "角色"))).strip() or "角色"
        duty = multi_agent_system_prompt(role)
        role_lines.append(f"- {name}: {duty}")
        role_bullets.append(f"- {name}")
    role_block = "\n".join(role_lines) if role_lines else "- （暂无角色）"
    role_intro = "\n".join(role_bullets) if role_bullets else "- （暂无角色）"

    discuss_prompt = (
        "你是“多Agent主控协调者”。当前阶段只做需求澄清，不允许启动多角色执行。\n"
        "请先准确理解用户原始表述的核心意图，再结合场景模板提供专业视角。\n"
        "当用户意图与模板默认方向不一致时，以用户意图为准（如用户要“写真PPT”是做展示，不是做策划方案）。\n"
        "请复述你理解的用户目标（忠实于用户原话的核心意图），"
        "再给 2-3 条有价值建议，最后提出最多 3 个澄清问题。\n"
        "如果信息已经足够执行，也不要直接执行，只需提示用户回复“确认需求”后立即启动多Agent执行。\n"
        "语气自然，不要像程序状态页。\n\n"
        f"场景澄清重点（提供专业框架，与用户意图结合使用）：{discuss_focus}\n\n"
        f"当前配置：场景={scenario_label}，模式={mode}，最大回合={rounds}\n"
        f"角色分工：\n{role_block}\n\n"
        f"累计议题：\n{pending_topic}\n\n"
        f"用户本轮消息：\n{user_message}\n"
    )
    output = await run_agent(
        message=discuss_prompt,
        session_id=f"{session_id}::ma::controller",
        config=config,
        on_stream=None,
        session=None,
        router=router,
        registry=registry,
        on_tool_call=None,
        on_tool_result=None,
        images=None,
        session_manager=None,
        session_store=None,
        memory_manager=None,
        extra_memory=extra_memory,
        trigger_event_id=trigger_event_id,
        trigger_text_preview=trigger_text_preview,
        group_compressor=None,
        multi_agent_internal=True,
    )
    output = output.strip()
    intro = (
        "我是主控Agent协调者。\n"
        f"当前场景：{scenario_label}；协作模式：{'并行' if mode == 'parallel' else '串行'}；"
        f"最大回合：{rounds}。\n"
        f"本轮将使用这 4 个角色协作：\n{role_intro}\n"
    )
    intro_block = f"{intro}\n" if include_intro else ""
    if output:
        return (
            f"{intro_block}{output}\n\n"
            "如果你觉得需求已经讲清楚，请回复“确认需求”；"
            "如果要改回合可回复“改为2轮”；不想继续可回复“取消”。\n"
            "你也可以直接指定交付物：例如“交付 PRD 文档 + 流程图图片 + 里程碑表”。"
        )
    return (
        f"{intro_block}"
        "我先和你确认需求：请补充目标、交付形式、截止时间与约束。"
        "信息齐后回复“确认需求”，我会立即启动多Agent执行。"
    )


def multi_agent_system_prompt(role: dict[str, object]) -> str:
    text = str(role.get("system_prompt", "")).strip()
    if text:
        return text
    return "负责从本角色视角提出可执行方案、风险与下一步建议。"


def compact_role_output(text: str, max_chars: int = 600) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def looks_like_bad_coordinator_output(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    hints = (
        "请把下方各角色多轮输出",
        "把各角色输出贴出来",
        "请粘贴各角色",
        "请提供各角色",
        "请先提供角色输出",
        "你提供的各角色多轮输出",
        "把下面这些内容贴出来",
        "你把材料发来",
        "请把材料发来",
        "我会把你提供的各角色",
    )
    if any(h in t for h in hints):
        return True
    ask_user_hints = (
        "请你补充",
        "请补充",
        "请回复",
        "你回复",
        "请回答",
        "你回答",
        "请确认",
        "需要你确认",
        "请发我",
        "你发我",
        "你把",
        "任选其一",
        "任意选一种",
        "为了我马上开工",
        "我就能继续",
        "我才能继续",
        "你要我交付",
        "你只要按",
        "回一条就行",
        "回一条即可",
        "需要你补",
        "你需要补",
        "你想要什么",
        "你想让我",
        "交付物类型：",
    )
    if any(h in t for h in ask_user_hints):
        return True
    if COORDINATOR_ASK_RE.search(t):
        return True
    if t.startswith("我将使用 ") and " 技能继续完成任务" in t:
        return True
    return "各角色" in t and ("贴出来" in t or "粘贴" in t or "发来" in t)


def _is_transient_multi_agent_error(exc: Exception) -> bool:
    text = str(exc).lower()
    signals = (
        "error 500",
        "server_error",
        "responses api error 500",
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    )
    return any(s in text for s in signals)


def looks_like_role_stall_output(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    hints = (
        "我将使用",
        "请你补充",
        "请补充",
        "你可以直接",
        "你现在想先做哪一块",
        "你先发",
        "我将以",
        "为了更高效推进",
        "为了马上进入状态",
        "请把要评审的内容发我",
    )
    if any(h in t for h in hints):
        return True
    return "?" in t or "？" in t


def _role_config(config: WhaleclawConfig, role_model: str) -> WhaleclawConfig:
    if not role_model:
        return config
    copied = config.model_copy(deep=True)
    copied.agent.model = role_model
    return copied


def need_image_output(user_message: str) -> bool:
    t = user_message.lower()
    negative_hints = (
        "不要图片",
        "不需要图片",
        "无需图片",
        "不要配图",
        "不需要配图",
        "无需配图",
        "纯文字",
        "仅文字",
        "只要文字",
        "text only",
    )
    if any(h in t for h in negative_hints):
        return False
    hints = (
        "配图",
        "图片",
        "图文",
        "海报",
        "封面",
        "插图",
        "image",
        "images",
        "poster",
        "cover",
    )
    return any(h in t for h in hints)


def extract_requested_deliverables(user_message: str) -> list[str]:
    t = user_message.lower()
    out: list[str] = []
    rules: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("word", ("word", "docx", ".docx"), ("不要word", "不需要word", "不要docx", "不需要docx")),
        ("ppt", ("ppt", "pptx", ".pptx", "幻灯片", "演示文稿"), ("不要ppt", "不需要ppt")),
        ("excel", ("excel", "xlsx", ".xlsx", "表格"), ("不要excel", "不需要excel", "不要表格")),
        ("html", ("html", "网页", "web页面"), ("不要html", "不需要html", "不要网页")),
        ("pdf", ("pdf", ".pdf"), ("不要pdf", "不需要pdf")),
        (
            "script",
            ("脚本", "python脚本", "py脚本", ".py", ".sh", "shell脚本", "bash脚本"),
            ("不要脚本", "不需要脚本"),
        ),
        (
            "image",
            ("图片", "配图", "图文", "海报", "封面", "image", "images"),
            ("不要图片", "不需要图片", "不要配图", "不需要配图"),
        ),
    ]
    for name, positives, negatives in rules:
        if any(n in t for n in negatives):
            continue
        if any(p in t for p in positives):
            out.append(name)
    return out


def _subset_registry(registry: ToolRegistry, allowed_names: set[str]) -> ToolRegistry:
    sub = ToolRegistry()
    for name in sorted(allowed_names):
        tool = registry.get(name)
        if tool is not None:
            sub.register(tool)
    return sub


def _choose_round_tool_lock(
    *,
    registry: ToolRegistry,
    user_message: str,
    shared_context: str,
    round_no: int,
    requested_deliverables: list[str],
) -> list[str]:
    query = f"第{round_no}轮\n{user_message}\n{shared_context}"
    selected = select_native_tool_names(registry, query)
    for deliverable in requested_deliverables:
        if deliverable == "word":
            selected.update({"file_write", "docx_edit", "bash"})
        elif deliverable == "ppt":
            selected.update({"file_write", "ppt_edit", "bash"})
        elif deliverable == "excel":
            selected.update({"file_write", "xlsx_edit", "bash"})
        elif deliverable == "html":
            selected.update({"file_write", "file_edit"})
        elif deliverable == "pdf":
            selected.update({"file_write", "bash"})
        elif deliverable == "image":
            selected.update({"browser", "file_write"})
    if not selected:
        selected = {"file_read", "file_write"}
    return sorted(selected)


def _clip_text_for_role_view(text: str, max_chars: int = 200) -> str:
    return " ".join(text.split()).strip() or "（无输出）"


def build_multi_agent_requirement_baseline(
    *,
    message: str,
    scenario: str,
    mode: str,
    max_rounds: int,
    requested_deliverables: list[str],
) -> str:
    scenario_label = MULTI_AGENT_SCENARIO_LABELS.get(scenario, scenario or "自定义")
    deliverables = ", ".join(requested_deliverables) if requested_deliverables else "未明确指定"
    text = (
        "【冻结需求基线（每轮必须遵守，不得改写目标）】\n"
        f"- 场景: {scenario_label}\n"
        f"- 协作: {mode}\n"
        f"- 轮次: {max_rounds}\n"
        f"- 交付类型: {deliverables}\n"
        f"- 用户确认需求原文:\n{message}\n"
    )
    return truncate_to_tokens(text, 700)


async def run_multi_agent_executor(
    *,
    message: str,
    session_id: str,
    config: WhaleclawConfig,
    on_stream: StreamCallback | None,
    router: ModelRouter,
    registry: ToolRegistry,
    images: list[ImageContent] | None,
    extra_memory: str,
    trigger_event_id: str,
    trigger_text_preview: str,
    ma_cfg: dict[str, object],
    on_round_result: OnRoundResult | None = None,
) -> str:
    roles = [
        role
        for role in cast(list[dict[str, object]], ma_cfg["roles"])
        if bool(role.get("enabled", True))
    ]
    if not roles:
        return "多Agent已启用，但没有可用角色。请在“多Agent”页面至少启用一个角色。"

    mode = cast(str, ma_cfg["mode"])
    max_rounds = cast(int, ma_cfg["max_rounds"])
    scenario = str(ma_cfg.get("scenario", "software_development")).strip()
    requested_deliverables = extract_requested_deliverables(message)
    _visual_file_types = {"ppt", "word", "html", "pdf"}
    include_image_output = (
        "image" in requested_deliverables
        or bool(_visual_file_types & set(requested_deliverables))
        or need_image_output(message)
    )
    requirement_baseline = build_multi_agent_requirement_baseline(
        message=message,
        scenario=scenario,
        mode=mode,
        max_rounds=max_rounds,
        requested_deliverables=requested_deliverables,
    )
    if on_stream is not None:
        await on_stream(
            f"[多Agent] 已启用 {len(roles)} 个角色，模式={mode}，最大回合={max_rounds}。"
        )

    history_blocks: list[str] = []
    round_deliveries: list[str] = []
    shared_context = ""
    prev_round_artifact_baseline: str = ""

    async def _run_one_role(
        role: dict[str, object],
        round_no: int,
        context_block: str,
        round_tools: list[str],
        round_registry: ToolRegistry,
        start_delay: float = 0.0,
    ) -> tuple[str, str, str]:
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        role_id = str(role.get("id", "role")).strip() or "role"
        role_name = str(role.get("name", role_id)).strip() or role_id
        prompt = multi_agent_system_prompt(role)
        role_msg = (
            f"你当前扮演的角色是「{role_name}」。\n"
            f"角色职责：{prompt}\n"
            f"当前是协作第 {round_no}/{max_rounds} 轮。\n"
            f"用户原始议题：{message}\n"
            "请结合角色专长和场景模板发表观点，但内容须紧扣用户原始议题的核心意图。\n"
        )
        role_msg += f"\n{requirement_baseline}\n"
        if context_block:
            role_msg += f"\n前序协作要点：\n{context_block}\n"
        if round_no > 1 and round_deliveries:
            prev_summary = (
                extract_round_delivery_section(round_deliveries[-1])
                or round_deliveries[-1]
            )
            prev_summary = truncate_to_tokens(prev_summary, 400)
            role_msg += (
                "\n上一轮（第 "
                f"{round_no - 1}"
                " 轮）交付内容摘要（仅供参考，本轮须重新制作，不要读取或修改上一轮文件）：\n"
                + prev_summary
                + "\n"
            )
        role_msg += (
            f"\n本轮工具锁定：{', '.join(round_tools)}\n"
            "\n请只输出本角色观点，必须总结为120字以内的精炼观点（严格不超过120字）。\n"
            "要求：给出明确判断和建议，用连贯段落表达，不要分点罗列，不要列步骤。\n"
            "限制：\n"
            "- 不要向用户追问或索取额外材料。\n"
            "- 信息不足时请自行做“最小必要假设”，并在输出中明确标注“假设”。\n"
            "- 禁止出现“我将使用xx技能继续完成任务”等过程话术。\n"
        )

        role_cfg = _role_config(config, str(role.get("model", "")).strip())
        last_error = ""
        for attempt in range(1, 4):
            try:
                output = await run_agent(
                    message=role_msg,
                    session_id=f"{session_id}::ma::{role_id}::r{round_no}",
                    config=role_cfg,
                    on_stream=None,
                    session=None,
                    router=router,
                    registry=round_registry,
                    on_tool_call=None,
                    on_tool_result=None,
                    images=None,
                    session_manager=None,
                    session_store=None,
                    memory_manager=None,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    group_compressor=None,
                    multi_agent_internal=True,
                )
                cleaned = output.strip()
                if not looks_like_role_stall_output(cleaned):
                    return role_id, role_name, cleaned
                rewrite_prompt = (
                    f"你是角色「{role_name}」。下面是一次无效输出，请改写成可直接交付的轮次结果。\n"
                    "要求：\n"
                    "1) 给出本轮结论（必须具体）\n"
                    "2) 给出3-5条可执行行动项（禁止提问）\n"
                    "3) 给出需要其他角色配合的点\n"
                    "4) 若信息不足，写明“假设”但仍要给出可执行版本\n\n"
                    f"无效输出：\n{cleaned}\n"
                )
                rewritten = await run_agent(
                    message=rewrite_prompt,
                    session_id=f"{session_id}::ma::{role_id}::rewrite::{round_no}",
                    config=role_cfg,
                    on_stream=None,
                    session=None,
                    router=router,
                    registry=round_registry,
                    on_tool_call=None,
                    on_tool_result=None,
                    images=None,
                    session_manager=None,
                    session_store=None,
                    memory_manager=None,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    group_compressor=None,
                    multi_agent_internal=True,
                )
                rewritten_clean = rewritten.strip()
                if not looks_like_role_stall_output(rewritten_clean):
                    return role_id, role_name, rewritten_clean
                fallback = "观点：在既有信息下先交付可用版本，假设最小化并保留下一轮优化空间。"
                return role_id, role_name, fallback
            except Exception as exc:
                last_error = str(exc)
                if not _is_transient_multi_agent_error(exc) or attempt >= 3:
                    break
                await asyncio.sleep(0.8 * attempt)
        return role_id, role_name, f"（角色执行失败，已跳过：{last_error or '未知错误'}）"

    aborted_round_no: int | None = None
    for round_no in range(1, max_rounds + 1):
        round_tools = _choose_round_tool_lock(
            registry=registry,
            user_message=message,
            shared_context=shared_context,
            round_no=round_no,
            requested_deliverables=requested_deliverables,
        )
        round_registry = _subset_registry(registry, set(round_tools))
        role_outputs: list[tuple[str, str, str]] = []
        if mode == "parallel":
            role_outputs = await asyncio.gather(
                *[
                    _run_one_role(
                        role,
                        round_no,
                        shared_context,
                        round_tools,
                        round_registry,
                        idx * 0.25,
                    )
                    for idx, role in enumerate(roles)
                ],
                return_exceptions=False,
            )
        else:
            for role in roles:
                role_outputs.append(
                    await _run_one_role(
                        role,
                        round_no,
                        shared_context,
                        round_tools,
                        round_registry,
                    )
                )

        lines: list[str] = [f"### 第 {round_no} 轮角色观点"]
        for _, role_name, out in role_outputs:
            result = _clip_text_for_role_view((out or "（无输出）").strip(), 200)
            lines.append(f"- {role_name}：{result}")
        role_round_block = "\n".join(lines)
        history_blocks.append(role_round_block)

        requested_text = ", ".join(requested_deliverables) if requested_deliverables else "未指定"
        _file_deliverable_types = {"ppt", "word", "excel", "html", "pdf"}
        requires_file_output = bool(
            _file_deliverable_types & set(requested_deliverables)
        )
        prev_round_block = ""
        if round_no > 1 and round_deliveries:
            prev_summary = (
                extract_round_delivery_section(round_deliveries[-1])
                or round_deliveries[-1]
            )
            prev_summary = truncate_to_tokens(prev_summary, 400)
            prev_round_block = (
                f"上一轮摘要（仅供参考，本轮须独立完成）：\n{prev_summary}\n"
            )
            if prev_round_artifact_baseline:
                prev_round_block += (
                    f"{prev_round_artifact_baseline}\n"
                    "本轮不得少于上轮，配图须重新搜索，布局和文案应有迭代。\n"
                )

        image_instruction = ""
        if include_image_output:
            image_instruction = (
                "配图流程：\n"
                "1) 先列配图清单（按页/模块），并声明「共 N 张配图」\n"
                "2) 再逐项调用 search_images 搜图下载，不同主体分别搜索，同一张图不要用于多页\n"
                "3) 搜完计划中的图后立刻进入写脚本步骤，禁止反复追加搜图\n"
            )

        round_synthesize_message: str = (
            "你是结果协调者。基于角色输出，独立完成本轮交付。\n"
            "【最高优先级】禁止向用户提问、要求补充材料或输出需求澄清模板。"
            "禁止输出执行计划或步骤预告（如「我先搜图再写脚本」「请稍等」），直接调用工具执行。"
            "你必须直接给出完整交付结果，信息不足时自行合理假设并执行。\n"
            "【语言】所有输出（文本、脚本中的文案、文件内容）的语言必须与用户议题的语言一致。\n"
            f"议题：{message}\n"
            f"场景：{MULTI_AGENT_SCENARIO_LABELS.get(scenario, scenario or '自定义')}\n"
            f"轮次：第 {round_no}/{max_rounds} 轮\n"
            f"{requirement_baseline}\n"
            f"工具锁定：{', '.join(round_tools)}\n"
            f"交付类型：{requested_text}\n"
            f"角色输出（仅供参考，必须以用户原始议题为准，角色观点偏离议题时忽略）：\n{role_round_block}\n\n"
            f"{prev_round_block}"
            f"{image_instruction}"
            "\n"
            f"## 第 {round_no} 轮交付（严格按此格式，每项必写）\n"
            "1. 本轮工具锁定（原样回写）\n"
            "2. 本轮可直接交付结果（必须给出具体内容）\n"
            "3. 配图（必须用 ![描述](绝对路径) 格式逐张列出所有已下载的图片）\n"
            "4. 与上一轮相比的改进点（首轮写「首轮基线」）\n"
            "\n"
            "规则：\n"
            "- 需要文件时必须调用工具生成并返回绝对路径，"
            f"文件名必须用 _V{round_no} 后缀（当前是第{round_no}轮，只能用 _V{round_no}）。\n"
            "- 脚本中图片路径硬编码绝对路径，禁止 os.environ。\n"
            "- PPT 插图直接用 slide.shapes.add_picture(path, left, top, width=w, height=h)，"
            "严禁对图片做任何预处理（禁止 PIL resize/crop/thumbnail，禁止 cv2 resize），"
            "系统会自动后处理人脸感知裁剪。\n"
            "- PPT 中文字必须在图片上方（先 add_picture 再 add_textbox），"
            "文本框之间不要位置重叠。\n"
            "- PPT 文本框必须设置 word_wrap=True，"
            "所有文本框的右边界不得超过幻灯片宽度。\n"
            "- PPT 布局：只有封面页可以用全屏大图做背景，内容页必须采用图文混排布局"
            "（如左图右文、上图下文、图占页面1/3~1/2），禁止每页都铺满整张图。\n"
            "- 不允许越做越差。\n"
        )
        round_delivery = ""
        delivery_accepted = False
        last_round_error = ""

        async def _run_round_coordinator(
            pass_no: int,
            base_message: str = round_synthesize_message,
            current_round_no: int = round_no,
            current_registry: ToolRegistry = round_registry,
        ) -> tuple[str, bool, str]:
            msg = base_message
            if pass_no == 2:
                msg += (
                    "\n这是本轮自动重跑（第2次也是最后一次）。"
                    "请严格围绕当前议题与上一轮交付输出最终结果；"
                    "禁止泛化模板与跑题内容。\n"
                )
            try:
                out = await run_agent(
                    message=msg,
                    session_id=f"{session_id}::ma::round::{current_round_no}::p{pass_no}",
                    config=config,
                    on_stream=None,
                    session=None,
                    router=router,
                    registry=current_registry,
                    on_tool_call=None,
                    on_tool_result=None,
                    images=None,
                    session_manager=None,
                    session_store=None,
                    memory_manager=None,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    group_compressor=None,
                    multi_agent_internal=True,
                )
                clean = out.strip()
                ok = bool(clean) and not looks_like_bad_coordinator_output(clean)
                return clean or out, ok, ""
            except Exception as exc:
                return "", False, str(exc)

        def _has_real_file_in_output(text: str) -> bool:
            paths = extract_delivery_artifact_paths(text, include_scripts=False)
            if any(Path(p).expanduser().exists() for p in paths):
                return True
            script_paths: list[str] = []
            for m in ABS_FILE_PATH_RE.finditer(text):
                p = m.group(1).strip()
                if not p:
                    continue
                suffix = Path(p).suffix.lower()
                if suffix in NON_DELIVERY_EXTS:
                    script_paths.append(p)
                    continue
                if Path(p).expanduser().exists():
                    return True
            for om in OFFICE_PATH_RE.finditer(text):
                op = om.group(1).strip()
                if op and Path(op).expanduser().exists():
                    return True
            for sp in script_paths:
                sp_path = Path(sp).expanduser()
                if not sp_path.exists() or sp_path.suffix.lower() != ".py":
                    continue
                try:
                    script_text = sp_path.read_text("utf-8", errors="ignore")[:8000]
                except Exception:
                    continue
                for sm in ABS_FILE_PATH_RE.finditer(script_text):
                    sp2 = sm.group(1).strip()
                    if not sp2:
                        continue
                    if Path(sp2).suffix.lower() in NON_DELIVERY_EXTS:
                        continue
                    if Path(sp2).expanduser().exists():
                        return True
            return False

        first_out, first_ok, first_err = await _run_round_coordinator(1)
        file_missing_on_first = False
        if first_ok:
            round_delivery = first_out
            delivery_accepted = True
            if requires_file_output and not _has_real_file_in_output(first_out):
                file_missing_on_first = True
                delivery_accepted = False
                last_round_error = "协调者未产出实际文件（只有文字或脚本，没有最终交付文件）"
                if on_stream is not None:
                    await on_stream(
                        f"[多Agent] 第 {round_no} 轮未检测到交付文件，自动重跑。"
                    )
        if not delivery_accepted and not first_ok:
            last_round_error = first_err
            if on_stream is not None:
                await on_stream(
                    f"[多Agent] 第 {round_no} 轮首次失败，自动重跑本轮一次（止损上限=2次）。"
                )
        if not delivery_accepted:
            retry_msg = round_synthesize_message
            if requires_file_output:
                retry_msg += (
                    "\n重要：上一次尝试未产出实际文件。本次你必须完成以下全部步骤，不可只输出文字就结束：\n"
                    "步骤1) search_images 搜图并下载"
                    "（按配图清单搜，搜完立刻进入步骤2，不要反复追加搜图）\n"
                    "步骤2) file_write 写完整 Python 生成脚本到 /tmp/"
                    "（图片路径硬编码为步骤1的绝对路径）\n"
                    "步骤3) bash 执行脚本产出最终文件（.pptx/.docx/.xlsx/.html/.pdf）\n"
                    "步骤4) 在交付文本中写明最终文件的绝对路径\n"
                    "关键：搜图完成后必须立刻写脚本执行，不要停留在搜图步骤。\n"
                )
            second_out, second_ok, second_err = await _run_round_coordinator(
                2, base_message=retry_msg
            )
            if second_ok:
                round_delivery = second_out
                delivery_accepted = True
                if requires_file_output and not _has_real_file_in_output(second_out):
                    delivery_accepted = False
            if not delivery_accepted:
                last_round_error = second_err or last_round_error
                best_text = second_out or (first_out if file_missing_on_first else "")
                if requires_file_output and best_text and not _has_real_file_in_output(best_text):
                    pass
                elif best_text:
                    round_delivery = best_text
                    delivery_accepted = True

        if not delivery_accepted:
            fallback_base = (
                f"## 第 {round_no} 轮交付\n"
                f"1) 本轮工具锁定：{', '.join(round_tools)}\n"
            )
            if round_no > 1 and round_deliveries:
                prev = extract_round_delivery_section(round_deliveries[-1]) or round_deliveries[-1]
                base = (
                    fallback_base
                    + "2) 本轮可直接交付结果（降级参考上一轮内容）：\n"
                    + prev
                    + "\n"
                    + "3) 结果协调者执行记录：本轮协调输出未达交付标准，"
                    + "已参考上一轮交付内容生成本轮基线。"
                )
                if last_round_error:
                    base += f" 错误摘要：{last_round_error[:180]}"
                base += "\n"
                base += "4) 与上一轮相比的改进点：本轮为稳定性兜底，未引入新改动。\n"
            else:
                base = (
                    fallback_base
                    + "2) 本轮可直接使用的产物：\n"
                    "- 基于角色观点的可执行摘要\n"
                    "- 当前轮次的风险与下一步建议\n"
                    "3) 结果协调者执行记录：未拿到稳定工具回执，已返回可直接使用文本版本。\n"
                    "4) 与上一轮相比的改进点：首轮基线。\n"
                )
            round_delivery = base
            aborted_round_no = round_no

        allow_script_paths = "script" in requested_deliverables
        detected_paths = extract_delivery_artifact_paths(
            round_delivery,
            include_scripts=allow_script_paths,
        )
        existing_paths = [
            str(Path(p).expanduser().resolve())
            for p in detected_paths
            if Path(p).expanduser().exists()
        ]
        existing_paths, rename_map = fix_version_suffix(existing_paths, round_no)
        if rename_map:
            for old_p, new_p in rename_map.items():
                round_delivery = round_delivery.replace(old_p, new_p)
        if existing_paths:
            verified_lines = [f"- {p}" for p in existing_paths]
            round_delivery = (
                f"{round_delivery.rstrip()}\n\n"
                "5) 系统校验的实际产物绝对路径（以此为准）\n"
                + "\n".join(verified_lines)
            )
        if existing_paths:
            snapshot_round_artifacts(existing_paths, round_no)
            prev_round_artifact_baseline = extract_artifact_baseline(existing_paths)
        round_deliveries.append(round_delivery)
        context_chunks: list[str] = []
        for idx, delivery in enumerate(round_deliveries, 1):
            section = extract_round_delivery_section(delivery) or delivery
            context_chunks.append(f"[第{idx}轮最终交付]\n{section.strip()}")
        shared_context = truncate_to_tokens(
            f"{requirement_baseline}\n\n" + "\n\n".join(context_chunks),
            1400,
        )
        display_delivery = f"{role_round_block}\n\n{round_delivery}"

        if on_round_result is not None:
            await on_round_result(round_no, display_delivery)

        if on_stream is not None:
            await on_stream(
                "[多Agent] "
                f"第 {round_no} 轮交付：\n"
                f"{display_delivery}\n"
            )

        if aborted_round_no == round_no:
            if on_stream is not None:
                await on_stream(
                    f"[多Agent] 第 {round_no} 轮连续两次失败，"
                    "已提前结束后续轮次以避免偏离目标与额外消耗。"
                )
            break

    if aborted_round_no is not None:
        return (
            f"多Agent执行已提前结束，停在第 {aborted_round_no} 轮（该轮两次失败已止损）。"
            f"已产出 {len(round_deliveries)} 轮结果。"
        )
    return f"多Agent执行完成，共 {len(round_deliveries)} 轮。"


# Backward-compatible aliases.
_run_multi_agent_controller_discussion = run_multi_agent_controller_discussion
_build_multi_agent_requirement_baseline = build_multi_agent_requirement_baseline
_run_multi_agent_executor = run_multi_agent_executor
_compact_role_output = compact_role_output


# Backward-compatible aliases for old imports.
_multi_agent_system_prompt = multi_agent_system_prompt
_looks_like_bad_coordinator_output = looks_like_bad_coordinator_output
_looks_like_role_stall_output = looks_like_role_stall_output
_need_image_output = need_image_output
_extract_requested_deliverables = extract_requested_deliverables
