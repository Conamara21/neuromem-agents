#!/usr/bin/env python3
"""Generate bilingual brain-inspired structure mindmaps for the README."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class BranchSpec:
    title: str
    summary: str
    module: str
    mechanism: str
    accent: str


@dataclass(frozen=True)
class DiagramSpec:
    title: str
    intro: str
    root_lines: tuple[str, ...]
    footer: str
    mermaid_heading: str
    mermaid_intro: str
    left_branches: tuple[BranchSpec, ...]
    right_branches: tuple[BranchSpec, ...]
    font_preference: tuple[str, ...]
    svg_output: Path
    png_output: Path
    mermaid_output: Path

    @property
    def all_branches(self) -> tuple[BranchSpec, ...]:
        return self.left_branches + self.right_branches


def _chunk_text(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]

    if " " in text:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
        if current:
            lines.append(current)
        return lines

    return [text[index:index + width] for index in range(0, len(text), width)]


def _format_module_lines(module: str) -> list[str]:
    if "/" not in module:
        return _chunk_text(module, 36)

    prefix, filename = module.rsplit("/", 1)
    lines = [prefix + "/"]
    lines.extend(_chunk_text(filename, 32))
    return lines


def _format_mechanism_lines(mechanism: str) -> list[str]:
    if ", " in mechanism:
        return mechanism.split(", ")
    return _chunk_text(mechanism, 32)


def _build_box_text(branch: BranchSpec) -> list[str]:
    lines = [branch.summary]
    lines.extend(_format_module_lines(branch.module))
    lines.extend(_format_mechanism_lines(branch.mechanism))
    return lines


def _render_text_block(
    x: float,
    y: float,
    lines: list[str],
    font_size: int,
    line_height: int,
    anchor: str = "start",
    fill: str = "#152033",
    weight: int = 500,
) -> str:
    spans: list[str] = []
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else str(line_height)
        spans.append(
            f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{font_size}" font-weight="{weight}" fill="{fill}">'
        + "".join(spans)
        + "</text>"
    )


def _load_font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _connector_path(start_x: float, start_y: float, end_x: float, end_y: float, side: str) -> str:
    bend = 135 if side == "right" else -135
    mid_x = start_x + bend
    pre_end_x = end_x - 85 if side == "right" else end_x + 85
    return (
        f"M {start_x:.1f} {start_y:.1f} "
        f"C {mid_x:.1f} {start_y:.1f}, {pre_end_x:.1f} {end_y:.1f}, {end_x:.1f} {end_y:.1f}"
    )


def render_svg(spec: DiagramSpec) -> None:
    width = 1820
    height = 1460
    root_x = width / 2
    root_y = height / 2
    root_rx = 190
    root_ry = 118
    box_w = 460
    box_h = 126
    box_gap = 26
    left_x = 120
    right_x = width - left_x - box_w

    total_left_height = len(spec.left_branches) * box_h + max(0, len(spec.left_branches) - 1) * box_gap
    total_right_height = len(spec.right_branches) * box_h + max(0, len(spec.right_branches) - 1) * box_gap
    left_start_y = (height - total_left_height) / 2
    right_start_y = (height - total_right_height) / 2

    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f"<title>{escape(spec.title)}</title>",
        f"<desc>{escape(spec.intro)}</desc>",
        "<defs>",
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#f8fafc"/>',
        '<stop offset="100%" stop-color="#eef4ff"/>',
        "</linearGradient>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#0f172a" flood-opacity="0.12"/>',
        "</filter>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>',
        (
            '<text x="90" y="86" font-size="36" font-weight="800" fill="#0f172a" '
            'font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">'
            f"{escape(spec.title)}</text>"
        ),
        (
            '<text x="90" y="126" font-size="19" font-weight="500" fill="#475569" '
            'font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">'
            f"{escape(spec.intro)}</text>"
        ),
        '<ellipse '
        f'cx="{root_x:.1f}" cy="{root_y:.1f}" rx="{root_rx}" ry="{root_ry}" '
        'fill="#0f172a" stroke="#1d4ed8" stroke-width="3" filter="url(#shadow)"/>',
    ]

    root_line_y = root_y - 36
    for index, line in enumerate(spec.root_lines):
        pieces.append(
            (
                f'<text x="{root_x:.1f}" y="{root_line_y + index * 34:.1f}" text-anchor="middle" '
                'font-size="28" font-weight="750" fill="#f8fafc" '
                'font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">'
                f"{escape(line)}</text>"
            )
        )

    def draw_side(branches: tuple[BranchSpec, ...], side: str, start_y: float, box_x: float) -> None:
        anchor_x = box_x + box_w if side == "left" else box_x
        box_text_x = box_x + 26
        title_x = box_text_x
        body_x = box_text_x
        start_x = root_x - root_rx if side == "left" else root_x + root_rx

        for index, branch in enumerate(branches):
            y = start_y + index * (box_h + box_gap)
            center_y = y + box_h / 2
            start_y_adjusted = root_y + (center_y - root_y) * 0.28
            pieces.append(
                f'<path d="{_connector_path(start_x, start_y_adjusted, anchor_x, center_y, "right" if side == "right" else "left")}" '
                f'fill="none" stroke="{branch.accent}" stroke-width="5" stroke-linecap="round"/>'
            )
            pieces.append(
                f'<circle cx="{anchor_x:.1f}" cy="{center_y:.1f}" r="8" fill="{branch.accent}"/>'
            )
            pieces.append(
                f'<rect x="{box_x:.1f}" y="{y:.1f}" width="{box_w}" height="{box_h}" rx="22" '
                f'fill="#ffffff" stroke="{branch.accent}" stroke-width="3" filter="url(#shadow)"/>'
            )
            pieces.append(
                _render_text_block(
                    title_x,
                    y + 34,
                    [branch.title],
                    font_size=24,
                    line_height=28,
                    fill="#0f172a",
                    weight=780,
                )
            )
            pieces.append(
                _render_text_block(
                    body_x,
                    y + 66,
                    _build_box_text(branch),
                    font_size=16,
                    line_height=22,
                    fill="#334155",
                    weight=520,
                )
            )

    draw_side(spec.left_branches, "left", left_start_y, left_x)
    draw_side(spec.right_branches, "right", right_start_y, right_x)

    pieces.append(
        (
            f'<text x="{width / 2:.1f}" y="{height - 36}" text-anchor="middle" font-size="16" '
            'font-weight="500" fill="#64748b" '
            'font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">'
            f"{escape(spec.footer)}</text>"
        )
    )
    pieces.append("</svg>")

    spec.svg_output.write_text("\n".join(pieces), encoding="utf-8")


def render_png(spec: DiagramSpec) -> None:
    width = 1820
    height = 1460
    root_x = width / 2
    root_y = height / 2
    root_rx = 190
    root_ry = 118
    box_w = 460
    box_h = 126
    box_gap = 26
    left_x = 120
    right_x = width - left_x - box_w

    title_font = _load_font(spec.font_preference, 36)
    intro_font = _load_font(spec.font_preference, 20)
    root_font = _load_font(spec.font_preference, 28)
    branch_title_font = _load_font(spec.font_preference, 24)
    branch_body_font = _load_font(spec.font_preference, 16)
    footer_font = _load_font(spec.font_preference, 16)

    image = Image.new("RGB", (width, height), "#f4f8ff")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(248 * (1 - ratio) + 238 * ratio)
        g = int(250 * (1 - ratio) + 244 * ratio)
        b = int(252 * (1 - ratio) + 255 * ratio)
        draw.line((0, y, width, y), fill=(r, g, b))

    draw.text((90, 52), spec.title, fill="#0f172a", font=title_font)
    draw.text((90, 98), spec.intro, fill="#475569", font=intro_font)

    shadow_offset = 8
    draw.ellipse(
        (
            root_x - root_rx + shadow_offset,
            root_y - root_ry + shadow_offset,
            root_x + root_rx + shadow_offset,
            root_y + root_ry + shadow_offset,
        ),
        fill="#dbe7ff",
    )
    draw.ellipse(
        (root_x - root_rx, root_y - root_ry, root_x + root_rx, root_y + root_ry),
        fill="#0f172a",
        outline="#1d4ed8",
        width=4,
    )

    root_line_y = root_y - 54
    for index, line in enumerate(spec.root_lines):
        bbox = draw.textbbox((0, 0), line, font=root_font)
        text_x = root_x - (bbox[2] - bbox[0]) / 2
        draw.text((text_x, root_line_y + index * 34), line, fill="#f8fafc", font=root_font)

    total_left_height = len(spec.left_branches) * box_h + max(0, len(spec.left_branches) - 1) * box_gap
    total_right_height = len(spec.right_branches) * box_h + max(0, len(spec.right_branches) - 1) * box_gap
    left_start_y = (height - total_left_height) / 2
    right_start_y = (height - total_right_height) / 2

    def draw_side(branches: tuple[BranchSpec, ...], side: str, start_y: float, box_x: float) -> None:
        anchor_x = box_x + box_w if side == "left" else box_x
        start_x = root_x - root_rx if side == "left" else root_x + root_rx
        body_x = box_x + 26

        for index, branch in enumerate(branches):
            y = start_y + index * (box_h + box_gap)
            center_y = y + box_h / 2
            start_y_adjusted = root_y + (center_y - root_y) * 0.28

            line_points = [
                (start_x, start_y_adjusted),
                (start_x + (135 if side == "right" else -135), start_y_adjusted),
                (anchor_x + (-85 if side == "right" else 85), center_y),
                (anchor_x, center_y),
            ]
            draw.line(line_points, fill=branch.accent, width=5, joint="curve")
            draw.ellipse((anchor_x - 8, center_y - 8, anchor_x + 8, center_y + 8), fill=branch.accent)

            shadow_box = (
                box_x + shadow_offset,
                y + shadow_offset,
                box_x + box_w + shadow_offset,
                y + box_h + shadow_offset,
            )
            draw.rounded_rectangle(shadow_box, radius=22, fill="#dde8f7")
            draw.rounded_rectangle(
                (box_x, y, box_x + box_w, y + box_h),
                radius=22,
                fill="#ffffff",
                outline=branch.accent,
                width=3,
            )
            draw.text((body_x, y + 18), branch.title, fill="#0f172a", font=branch_title_font)

            text_lines = _build_box_text(branch)
            for line_index, line in enumerate(text_lines):
                draw.text(
                    (body_x, y + 52 + line_index * 22),
                    line,
                    fill="#334155",
                    font=branch_body_font,
                )

    draw_side(spec.left_branches, "left", left_start_y, left_x)
    draw_side(spec.right_branches, "right", right_start_y, right_x)

    footer_bbox = draw.textbbox((0, 0), spec.footer, font=footer_font)
    footer_x = (width - (footer_bbox[2] - footer_bbox[0])) / 2
    draw.text((footer_x, height - 42), spec.footer, fill="#64748b", font=footer_font)

    image.save(spec.png_output)


def render_mermaid(spec: DiagramSpec) -> None:
    lines = [
        f"# {spec.mermaid_heading}",
        "",
        spec.mermaid_intro,
        "",
        "```mermaid",
        "mindmap",
        f"  root(({spec.root_lines[0]} {spec.root_lines[1]}))",
    ]

    for branch in spec.all_branches:
        lines.append(f"    {branch.title}")
        lines.append(f"      {branch.summary}")
        lines.append(f"      module {branch.module}")
        lines.append(f"      key {branch.mechanism}")

    lines.extend(["```", ""])
    spec.mermaid_output.write_text("\n".join(lines), encoding="utf-8")


def build_specs(repo_root: Path) -> tuple[DiagramSpec, DiagramSpec]:
    chinese = DiagramSpec(
        title="NeuroMem 脑启发结构图",
        intro="按器官级功能映射当前记忆架构的研究型实现，并标注对应代码模块。",
        root_lines=("NeuroMem", "脑启发记忆结构", "代码映射图"),
        footer="Generated from scripts/render_brain_region_mindmap.py",
        mermaid_heading="NeuroMem 脑启发结构思维导图",
        mermaid_intro="此文件由 `scripts/render_brain_region_mindmap.py` 生成，建议编辑生成脚本而不是手改本文件。",
        left_branches=(
            BranchSpec(
                title="海马体 Hippocampus",
                summary="工作记忆缓冲 + 记忆巩固",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="working_memory_buffer, consolidate()",
                accent="#2563eb",
            ),
            BranchSpec(
                title="大脑皮质 Cortex",
                summary="长期记忆存储 + 知识沉淀",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="long_term_memory",
                accent="#16a34a",
            ),
            BranchSpec(
                title="边缘系统 Limbic System",
                summary="重要性标注 + 检索偏置",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="importance_score, attention_weight",
                accent="#ea580c",
            ),
            BranchSpec(
                title="神经振荡 Neural Oscillations",
                summary="脉冲放电 + 节律同步",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="SpikingNeuralNetwork",
                accent="#7c3aed",
            ),
            BranchSpec(
                title="突触可塑性 Synaptic Plasticity",
                summary="时序依赖的连接更新",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="STDPMechanism",
                accent="#dc2626",
            ),
        ),
        right_branches=(
            BranchSpec(
                title="前额叶皮质 Prefrontal Cortex",
                summary="执行控制 + 注意力门控",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="AttentionGate",
                accent="#0f766e",
            ),
            BranchSpec(
                title="小脑 Cerebellum",
                summary="元学习调参 + 协调优化",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="MetaLearningController",
                accent="#0891b2",
            ),
            BranchSpec(
                title="颞叶 Temporal Lobe",
                summary="语言模式预测 + 状态外推",
                module="neuromem/core/hierarchical_memory.py",
                mechanism="PredictionEngine",
                accent="#ca8a04",
            ),
            BranchSpec(
                title="顶叶 Parietal Lobe",
                summary="跨脑区协调 + 注意定向",
                module="neuromem/core/hierarchical_memory.py",
                mechanism="coordinate_regions()",
                accent="#9333ea",
            ),
            BranchSpec(
                title="预测编码 Predictive Coding",
                summary="预测误差驱动存储",
                module="neuromem/core/hierarchical_memory.py",
                mechanism="_evaluate_storage_necessity()",
                accent="#be123c",
            ),
        ),
        font_preference=(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ),
        svg_output=repo_root / "docs" / "neuromem_brain_regions_mindmap.svg",
        png_output=repo_root / "docs" / "neuromem_brain_regions_mindmap.png",
        mermaid_output=repo_root / "docs" / "neuromem_brain_regions_mindmap.md",
    )

    english = DiagramSpec(
        title="NeuroMem Brain-Inspired Structure Map",
        intro="A code-backed map of the research-oriented memory components framed as organ-level analogies.",
        root_lines=("NeuroMem", "Brain-Inspired Memory", "Code Map"),
        footer="Generated from scripts/render_brain_region_mindmap.py",
        mermaid_heading="NeuroMem Brain-Inspired Structure Mindmap",
        mermaid_intro="This file is generated by `scripts/render_brain_region_mindmap.py`. Edit the generator instead of editing this file directly.",
        left_branches=(
            BranchSpec(
                title="Hippocampus",
                summary="Working buffer + consolidation",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="working_memory_buffer, consolidate()",
                accent="#2563eb",
            ),
            BranchSpec(
                title="Cortex",
                summary="Long-term storage + knowledge retention",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="long_term_memory",
                accent="#16a34a",
            ),
            BranchSpec(
                title="Limbic System",
                summary="Importance tagging + retrieval bias",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="importance_score, attention_weight",
                accent="#ea580c",
            ),
            BranchSpec(
                title="Neural Oscillations",
                summary="Spike dynamics + rhythm binding",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="SpikingNeuralNetwork",
                accent="#7c3aed",
            ),
            BranchSpec(
                title="Synaptic Plasticity",
                summary="Timing-based connection updates",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="STDPMechanism",
                accent="#dc2626",
            ),
        ),
        right_branches=(
            BranchSpec(
                title="Prefrontal Cortex",
                summary="Executive control + attention gating",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="AttentionGate",
                accent="#0f766e",
            ),
            BranchSpec(
                title="Cerebellum",
                summary="Meta-learning + parameter tuning",
                module="neuromem/core/enhanced_memory_manager.py",
                mechanism="MetaLearningController",
                accent="#0891b2",
            ),
            BranchSpec(
                title="Temporal Lobe",
                summary="Language prediction + state extrapolation",
                module="neuromem/core/hierarchical_memory.py",
                mechanism="PredictionEngine",
                accent="#ca8a04",
            ),
            BranchSpec(
                title="Parietal Lobe",
                summary="Cross-region coordination + attention steering",
                module="neuromem/core/hierarchical_memory.py",
                mechanism="coordinate_regions()",
                accent="#9333ea",
            ),
            BranchSpec(
                title="Predictive Coding",
                summary="Prediction error decides storage",
                module="neuromem/core/hierarchical_memory.py",
                mechanism="_evaluate_storage_necessity()",
                accent="#be123c",
            ),
        ),
        font_preference=(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        svg_output=repo_root / "docs" / "neuromem_brain_regions_mindmap_en.svg",
        png_output=repo_root / "docs" / "neuromem_brain_regions_mindmap_en.png",
        mermaid_output=repo_root / "docs" / "neuromem_brain_regions_mindmap_en.md",
    )

    return chinese, english


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    specs = build_specs(repo_root)
    for spec in specs:
        render_svg(spec)
        render_png(spec)
        render_mermaid(spec)
        print(f"generated {spec.svg_output.relative_to(repo_root)}")
        print(f"generated {spec.png_output.relative_to(repo_root)}")
        print(f"generated {spec.mermaid_output.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
