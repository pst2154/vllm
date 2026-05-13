#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


VARIANT_ORDER = [
    "vanilla_nvfp4",
    "mtp_nvfp4",
    "dflash_nvfp4",
    "optimized_nvfp4",
]

VARIANT_LABELS = {
    "vanilla_nvfp4": "Vanilla NVFP4",
    "mtp_nvfp4": "Native MTP",
    "dflash_nvfp4": "DFlash k=15",
    "optimized_nvfp4": "DFlash k=15 + GDN T16",
}

VARIANT_NOTES = {
    "vanilla_nvfp4": "Target model only, no speculative decoding.",
    "mtp_nvfp4": "Native MTP speculation with num_speculative_tokens=1.",
    "dflash_nvfp4": "DFlash draft model with num_speculative_tokens=15.",
    "optimized_nvfp4": (
        "DFlash k=15 plus fp16 GDN SSM cache and the Qwen GDN T16 commit1 "
        "unpaired Triton verifier path."
    ),
}

COLORS = {
    "vanilla_nvfp4": "#4b5563",
    "mtp_nvfp4": "#2563eb",
    "dflash_nvfp4": "#059669",
    "optimized_nvfp4": "#dc2626",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def finite_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def parse_result(path: Path) -> dict[str, Any]:
    match = re.match(r"(.+)_c([0-9]+)_tg128\.json$", path.name)
    if not match or path.name.endswith("_prewarm.json"):
        raise ValueError(f"unexpected result filename: {path.name}")
    variant = match.group(1)
    concurrency = int(match.group(2))
    data = load_json(path)
    benches = data.get("benchmarks")
    if not isinstance(benches, list) or not benches:
        raise ValueError(f"{path} missing benchmarks")
    bench = benches[0]
    if not isinstance(bench, dict):
        raise ValueError(f"{path} benchmark is not an object")
    tg = bench.get("tg_throughput")
    if not isinstance(tg, dict):
        raise ValueError(f"{path} missing tg_throughput")
    values = [
        float(item)
        for item in tg.get("values", [])
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]
    contract = data.get("qwen_tg128_contract")
    if not isinstance(contract, dict):
        contract = {}
    return {
        "variant": variant,
        "label": VARIANT_LABELS.get(variant, variant),
        "concurrency": concurrency,
        "mean_tps": finite_float(tg.get("mean")),
        "std_tps": finite_float(tg.get("std")),
        "runs": int(contract.get("run_count") or len(values)),
        "values": values,
        "prompt_tokens": bench.get("prompt_size"),
        "generate_tokens": bench.get("response_size"),
        "depth": bench.get("context_size"),
        "result_file": path.name,
        "timestamp": data.get("timestamp", ""),
        "candidate_name": contract.get("candidate_name", ""),
    }


def variant_sort_key(variant: str) -> int:
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def format_tps(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.2f}"


def format_gain(value: float, base: float) -> str:
    if not math.isfinite(value) or not math.isfinite(base) or base == 0:
        return ""
    delta = value - base
    pct = delta / base * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f} / {sign}{pct:.1f}%"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "variant",
        "label",
        "concurrency",
        "mean_tps",
        "std_tps",
        "runs",
        "values",
        "result_file",
        "timestamp",
        "candidate_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["values"] = " ".join(format_tps(v) for v in row["values"])
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def matrix(rows: list[dict[str, Any]]) -> tuple[list[str], list[int], dict[tuple[str, int], dict[str, Any]]]:
    variants = sorted({row["variant"] for row in rows}, key=variant_sort_key)
    concurrencies = sorted({int(row["concurrency"]) for row in rows})
    by_key = {(row["variant"], int(row["concurrency"])): row for row in rows}
    return variants, concurrencies, by_key


def write_markdown(rows: list[dict[str, Any]], path: Path, run_id: str) -> None:
    variants, concurrencies, by_key = matrix(rows)
    vanilla_by_c = {
        c: by_key.get(("vanilla_nvfp4", c), {}).get("mean_tps", math.nan)
        for c in concurrencies
    }

    lines: list[str] = []
    lines.append(f"# Qwen3.6 NVFP4 TG128 Progression Sweep")
    lines.append("")
    lines.append(f"Run ID: `{run_id}`")
    lines.append("")
    lines.append("Benchmark shape: `pp=2048`, `tg=128`, `depth=0`, no cache, generation latency.")
    lines.append("Each cell uses 2 explicit warmup requests and 10 measured requests. llama-benchy internal warmup, coherence check, and prompt adaptation were disabled for this sweep.")
    lines.append("")
    lines.append("![TG128 progression](qwen36_progression_tg128.svg)")
    lines.append("")
    lines.append("## Throughput Matrix")
    lines.append("")
    header = ["Variant"] + [f"c{c}" for c in concurrencies]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:" for _ in concurrencies]) + " |")
    for variant in variants:
        row = [VARIANT_LABELS.get(variant, variant)]
        for concurrency in concurrencies:
            item = by_key.get((variant, concurrency))
            row.append(format_tps(item["mean_tps"]) if item else "")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Gain Versus Vanilla")
    lines.append("")
    lines.append("| Variant | " + " | ".join(f"c{c}" for c in concurrencies) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in concurrencies) + " |")
    for variant in variants:
        row = [VARIANT_LABELS.get(variant, variant)]
        for concurrency in concurrencies:
            item = by_key.get((variant, concurrency))
            base = vanilla_by_c.get(concurrency, math.nan)
            row.append(format_gain(item["mean_tps"], base) if item else "")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## c1 Progression")
    lines.append("")
    lines.append("| Stage | Avg TG128 TPS | Gain vs previous | Gain vs vanilla | Notes |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    previous = math.nan
    vanilla = vanilla_by_c.get(1, math.nan)
    for variant in variants:
        item = by_key.get((variant, 1))
        if not item:
            continue
        value = item["mean_tps"]
        gain_prev = "baseline" if not math.isfinite(previous) else format_gain(value, previous)
        gain_vanilla = "baseline" if variant == "vanilla_nvfp4" else format_gain(value, vanilla)
        lines.append(
            "| "
            + " | ".join(
                [
                    VARIANT_LABELS.get(variant, variant),
                    format_tps(value),
                    gain_prev,
                    gain_vanilla,
                    VARIANT_NOTES.get(variant, ""),
                ]
            )
            + " |"
        )
        previous = value

    lines.append("")
    lines.append("## Raw Result Files")
    lines.append("")
    lines.append("| Variant | Concurrency | Runs | File |")
    lines.append("| --- | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| {VARIANT_LABELS.get(row['variant'], row['variant'])} | "
            f"{row['concurrency']} | {row['runs']} | "
            f"`raw/{row['result_file']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg(rows: list[dict[str, Any]], path: Path) -> None:
    variants, concurrencies, by_key = matrix(rows)
    width = 1040
    height = 620
    left = 90
    right = 250
    top = 60
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = [row["mean_tps"] for row in rows if math.isfinite(row["mean_tps"])]
    max_y = max(values) if values else 1.0
    min_y = min(values) if values else 0.0
    y0 = max(0.0, math.floor(min_y / 10.0) * 10.0 - 10.0)
    y1 = math.ceil(max_y / 10.0) * 10.0 + 10.0
    if y1 <= y0:
        y1 = y0 + 10.0

    def x_pos(concurrency: int) -> float:
        if len(concurrencies) == 1:
            return left + plot_w / 2.0
        idx = concurrencies.index(concurrency)
        return left + idx * plot_w / (len(concurrencies) - 1)

    def y_pos(value: float) -> float:
        return top + (y1 - value) / (y1 - y0) * plot_h

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Qwen TG128 throughput progression">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append('<text x="90" y="34" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">Qwen3.6 NVFP4 TG128 Progression on DGX Spark</text>')
    parts.append('<text x="90" y="56" font-family="Inter, Arial, sans-serif" font-size="13" fill="#4b5563">2 warmup requests, 10 measured requests, pp=2048, tg=128, no cache</text>')

    for step in range(6):
        value = y0 + (y1 - y0) * step / 5.0
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="#6b7280">{value:.0f}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>')

    for concurrency in concurrencies:
        x = x_pos(concurrency)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#111827" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#111827">c{concurrency}</text>')

    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#374151">Concurrency</text>')
    parts.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#374151">Average TG128 tokens/sec</text>')

    for variant in variants:
        points = []
        for concurrency in concurrencies:
            row = by_key.get((variant, concurrency))
            if row and math.isfinite(row["mean_tps"]):
                points.append((x_pos(concurrency), y_pos(row["mean_tps"]), row["mean_tps"]))
        if not points:
            continue
        color = COLORS.get(variant, "#111827")
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
            parts.append(f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{color}">{value:.1f}</text>')

    legend_x = left + plot_w + 42
    legend_y = top + 22
    parts.append(f'<text x="{legend_x}" y="{legend_y - 20}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">Variants</text>')
    for idx, variant in enumerate(variants):
        y = legend_y + idx * 30
        color = COLORS.get(variant, "#111827")
        label = html.escape(VARIANT_LABELS.get(variant, variant))
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{y + 5}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#111827">{label}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_json(rows: list[dict[str, Any]], path: Path, run_id: str) -> None:
    variants, concurrencies, by_key = matrix(rows)
    payload = {
        "run_id": run_id,
        "prompt_tokens": 2048,
        "generate_tokens": 128,
        "depth": 0,
        "prewarm_runs": 2,
        "measured_runs": 10,
        "variants": variants,
        "concurrencies": concurrencies,
        "results": rows,
    }
    if ("vanilla_nvfp4", 1) in by_key and ("optimized_nvfp4", 1) in by_key:
        base = by_key[("vanilla_nvfp4", 1)]["mean_tps"]
        opt = by_key[("optimized_nvfp4", 1)]["mean_tps"]
        payload["c1_optimized_gain_vs_vanilla_tps"] = opt - base
        payload["c1_optimized_gain_vs_vanilla_pct"] = (opt - base) / base * 100.0
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*_c*_tg128.json")):
        if path.name.endswith("_prewarm.json"):
            continue
        rows.append(parse_result(path))
    rows.sort(key=lambda row: (variant_sort_key(row["variant"]), int(row["concurrency"])))

    if not rows:
        raise SystemExit(f"no result JSONs found in {raw_dir}")

    write_csv(rows, output_dir / "qwen36_progression_tg128.csv")
    write_markdown(rows, output_dir / "qwen36_progression_tg128.md", args.run_id)
    write_svg(rows, output_dir / "qwen36_progression_tg128.svg")
    write_summary_json(rows, output_dir / "qwen36_progression_tg128.summary.json", args.run_id)

    c1_values = [
        row["mean_tps"] for row in rows if int(row["concurrency"]) == 1
    ]
    if c1_values:
        print(f"qwen36_progression_c1_mean_of_variants={mean(c1_values):.6f}")
    print(f"qwen36_progression_rows={len(rows)}")
    print(f"qwen36_progression_output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
