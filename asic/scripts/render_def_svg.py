#!/usr/bin/env python3
"""Render an OpenROAD DEF into readable SVG debug views.

This is not a mask-accurate GDS viewer. It is a design-review renderer:
it highlights die/core, logic density, pins, power straps, and a sampled
set of routed signal wires so a placed/routed chip is visually inspectable.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
from collections import defaultdict


PT = re.compile(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)")
PLACED = re.compile(r"\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")
COMP = re.compile(r"^\s*-\s+(\S+)\s+(\S+).*?\+\s+(?:SOURCE\s+\S+\s+)?PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")
ROW = re.compile(r"^ROW\s+\S+\s+\S+\s+(-?\d+)\s+(-?\d+).*?DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(-?\d+)\s+(-?\d+)")
LAYER_SEG = re.compile(r"(?:\+ ROUTED|NEW)\s+(\S+)\s+(\d+)?(?:\s+\+ SHAPE\s+\S+)?\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(\*|-?\d+)\s+(\*|-?\d+)\s*\)")
SIGNAL_SEG = re.compile(r"(?:\+ ROUTED|NEW)\s+(\S+)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(\*|-?\d+)\s+(\*|-?\d+)\s*\)")


COLORS = {
    "li1": "#8fd3ff",
    "met1": "#42b883",
    "met2": "#ffd166",
    "met3": "#ef476f",
    "met4": "#7b61ff",
    "met5": "#06d6a0",
}


def parse_def(path: str) -> dict:
    data = {
        "die": (0, 0, 0, 0),
        "rows": [],
        "components": [],
        "pins": [],
        "special_routes": [],
        "signal_routes": [],
    }
    section = None
    signal_limit = 30000
    signal_seen = 0
    current_pin = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("DIEAREA"):
                pts = PT.findall(line)
                if len(pts) >= 2:
                    data["die"] = tuple(map(int, (*pts[0], *pts[1])))
            elif line.startswith("ROW "):
                m = ROW.search(line)
                if m:
                    x, y, do, by, sx, sy = map(int, m.groups())
                    data["rows"].append((x, y, do, by, sx, sy))
            elif line.startswith("COMPONENTS"):
                section = "components"
            elif line.startswith("PINS"):
                section = "pins"
            elif line.startswith("SPECIALNETS"):
                section = "special"
            elif line.startswith("NETS"):
                section = "nets"
            elif line.startswith("END "):
                if "COMPONENTS" in line or "PINS" in line or "SPECIALNETS" in line or "NETS" in line:
                    section = None

            if section == "components":
                m = COMP.search(line)
                if m:
                    name, cell, x, y = m.groups()
                    if name.startswith("FILLER"):
                        kind = "filler"
                    elif "tap" in cell or "decap" in cell or "fill" in cell or "endcap" in cell:
                        kind = "physical"
                    elif name.startswith("ANTENNA"):
                        kind = "antenna"
                    else:
                        kind = "logic"
                    data["components"].append((int(x), int(y), kind))

            elif section == "pins":
                if line.lstrip().startswith("- "):
                    current_pin = line.split()[1]
                m = PLACED.search(line)
                if m and current_pin:
                    data["pins"].append((current_pin, int(m.group(1)), int(m.group(2))))

            elif section == "special":
                m = LAYER_SEG.search(line)
                if m:
                    layer, width, x1, y1, x2, y2 = m.groups()
                    x1, y1 = int(x1), int(y1)
                    x2 = x1 if x2 == "*" else int(x2)
                    y2 = y1 if y2 == "*" else int(y2)
                    data["special_routes"].append((layer, int(width or 0), x1, y1, x2, y2))

            elif section == "nets" and signal_seen < signal_limit:
                m = SIGNAL_SEG.search(line)
                if m:
                    layer, x1, y1, x2, y2 = m.groups()
                    x1, y1 = int(x1), int(y1)
                    x2 = x1 if x2 == "*" else int(x2)
                    y2 = y1 if y2 == "*" else int(y2)
                    if abs(x2 - x1) + abs(y2 - y1) > 0:
                        data["signal_routes"].append((layer, x1, y1, x2, y2))
                        signal_seen += 1
    return data


def map_pt(x: int, y: int, bounds: tuple[int, int, int, int], margin: int, scale: float, height: float) -> tuple[float, float]:
    x0, y0, _, _ = bounds
    return margin + (x - x0) * scale, margin + height - (y - y0) * scale


def rect_svg(x1, y1, x2, y2, bounds, margin, scale, height, attrs):
    sx1, sy1 = map_pt(x1, y1, bounds, margin, scale, height)
    sx2, sy2 = map_pt(x2, y2, bounds, margin, scale, height)
    x, y = min(sx1, sx2), min(sy1, sy2)
    w, h = abs(sx2 - sx1), abs(sy2 - sy1)
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" {attrs}/>'


def line_svg(x1, y1, x2, y2, bounds, margin, scale, height, attrs):
    sx1, sy1 = map_pt(x1, y1, bounds, margin, scale, height)
    sx2, sy2 = map_pt(x2, y2, bounds, margin, scale, height)
    return f'<line x1="{sx1:.2f}" y1="{sy1:.2f}" x2="{sx2:.2f}" y2="{sy2:.2f}" {attrs}/>'


def render(data: dict, out: str, title: str, bounds: tuple[int, int, int, int], route_stride: int = 10) -> None:
    x0, y0, x1, y1 = bounds
    bw, bh = x1 - x0, y1 - y0
    margin = 70
    width = 1400
    scale = (width - margin * 2) / bw
    height = bh * scale
    full_h = height + margin * 2 + 95
    svg = []

    def add(s: str) -> None:
        svg.append(s)

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{full_h:.0f}" viewBox="0 0 {width} {full_h:.0f}">')
    add('<rect width="100%" height="100%" fill="#101217"/>')
    add(f'<text x="30" y="38" fill="#f5f7fb" font-size="24" font-family="Menlo,Consolas,monospace">{html.escape(title)}</text>')
    add(f'<text x="30" y="{full_h - 28:.0f}" fill="#b7becf" font-size="14" font-family="Menlo,Consolas,monospace">yellow/orange=logic density, gray=physical/filler density, cyan/green/pink/purple=sampled signal routes, thick blue/teal=power grid, dots=edge pins</text>')
    add(rect_svg(x0, y0, x1, y1, bounds, margin, scale, height, 'fill="#151b24" stroke="#e5e7eb" stroke-width="2"'))

    # Standard-cell row guides.
    for rx, ry, do, by, sx, sy in data["rows"][::4]:
        rx2 = rx + do * sx
        if y0 <= ry <= y1:
            add(line_svg(max(rx, x0), ry, min(rx2, x1), ry, bounds, margin, scale, height, 'stroke="#2f3745" stroke-width="0.55" opacity="0.55"'))

    # Density tiles keep the SVG readable even with hundreds of thousands of cells.
    bins = 90
    logic = defaultdict(int)
    physical = defaultdict(int)
    antenna = []
    for cx, cy, kind in data["components"]:
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        bx = min(bins - 1, max(0, int((cx - x0) / bw * bins)))
        by = min(bins - 1, max(0, int((cy - y0) / bh * bins)))
        if kind == "logic":
            logic[(bx, by)] += 1
        elif kind == "antenna":
            antenna.append((cx, cy))
        else:
            physical[(bx, by)] += 1

    max_logic = max(logic.values(), default=1)
    max_phys = max(physical.values(), default=1)
    tile_w = bw / bins
    tile_h = bh / bins
    for grid, max_v, color in ((physical, max_phys, (130, 139, 154)), (logic, max_logic, (255, 177, 66))):
        for (bx, by), count in grid.items():
            alpha = 0.08 + 0.72 * math.sqrt(count / max_v)
            r, g, b = color
            attrs = f'fill="rgb({r},{g},{b})" opacity="{alpha:.3f}"'
            add(rect_svg(x0 + bx * tile_w, y0 + by * tile_h, x0 + (bx + 1) * tile_w, y0 + (by + 1) * tile_h, bounds, margin, scale, height, attrs))

    # Power grid, drawn before signal routes but with heavier strokes.
    for layer, width_db, ax, ay, bx, by in data["special_routes"]:
        if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
            continue
        color = "#36c2ff" if layer in ("met4", "met2") else "#35f0a3"
        sw = max(0.8, min(4.2, (width_db or 900) * scale))
        add(line_svg(ax, ay, bx, by, bounds, margin, scale, height, f'stroke="{color}" stroke-width="{sw:.2f}" opacity="0.5" stroke-linecap="round"'))

    # Sampled signal routes. Full route SVG would be too visually busy and too large.
    for i, (layer, ax, ay, bx, by) in enumerate(data["signal_routes"]):
        if i % route_stride:
            continue
        if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
            continue
        color = COLORS.get(layer, "#ffffff")
        add(line_svg(ax, ay, bx, by, bounds, margin, scale, height, f'stroke="{color}" stroke-width="0.9" opacity="0.32" stroke-linecap="round"'))

    for name, px, py in data["pins"]:
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        sx, sy = map_pt(px, py, bounds, margin, scale, height)
        add(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3.2" fill="#f5f7fb" opacity="0.9"><title>{html.escape(name)}</title></circle>')

    for ax, ay in antenna:
        sx, sy = map_pt(ax, ay, bounds, margin, scale, height)
        add(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="1.7" fill="#ff4d6d" opacity="0.85"/>')

    add("</svg>")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def component_bounds(data: dict, kind: str = "logic") -> tuple[int, int, int, int]:
    pts = [(x, y) for x, y, k in data["components"] if k == kind]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 45000
    die = data["die"]
    return (
        max(die[0], min(xs) - pad),
        max(die[1], min(ys) - pad),
        min(die[2], max(xs) + pad),
        min(die[3], max(ys) + pad),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("def_file")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data = parse_def(args.def_file)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(args.def_file)), "render")
    die = data["die"]
    core = component_bounds(data, "logic")
    # A tighter center crop around the arithmetic/control cluster.
    cx = (core[0] + core[2]) // 2
    cy = (core[1] + core[3]) // 2
    span = min(max(core[2] - core[0], core[3] - core[1]), 430000)
    zoom = (max(die[0], cx - span // 2), max(die[1], cy - span // 2), min(die[2], cx + span // 2), min(die[3], cy + span // 2))

    base = os.path.splitext(os.path.basename(args.def_file))[0]
    title = base.replace("_", " ")
    render(data, os.path.join(out_dir, f"{base}.overview.svg"), f"{title} - full die", die, route_stride=16)
    render(data, os.path.join(out_dir, f"{base}.core.svg"), f"{title} - standard-cell core", core, route_stride=8)
    render(data, os.path.join(out_dir, f"{base}.routes_zoom.svg"), f"{title} - routed logic zoom", zoom, route_stride=2)
    print(f"Wrote SVG renders to {out_dir}")


if __name__ == "__main__":
    main()
