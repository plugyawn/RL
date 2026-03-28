#!/usr/bin/env python3
"""Summarize GRPO rollout stability signals from TensorBoard logs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

SIZE_GUIDANCE = {
    event_accumulator.TENSORS: 0,
    event_accumulator.SCALARS: 0,
}

SUMMARY_METRICS = [
    "validation/accuracy",
    "train/reward",
    "train/loss",
    "train/truncation_rate",
    "train/natural_termination_rate",
    "train/mean_gen_tokens_per_sample",
    "train/max_gen_tokens_per_sample",
    "train/gen_kl_error",
    "train/policy_kl_error",
    "train/kl_penalty",
    "train/dg_gate_mean",
    "train/dg_gate_spread",
    "timing/setup/total_setup_time_s",
    "timing/train/total_step_time",
]


def parse_run_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    label, run_dir = value.split("=", 1)
    return label, run_dir


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    event_files = sorted(glob.glob(os.path.join(run_dir, "**", "events*tfevents*"), recursive=True))
    merged: dict[str, dict[int, float]] = {}
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            event_file, size_guidance=SIZE_GUIDANCE
        )
        accumulator.Reload()
        for metric_name in accumulator.scalars.Keys():
            metric_steps = merged.setdefault(metric_name, {})
            for scalar in accumulator.Scalars(metric_name):
                metric_steps[scalar.step] = scalar.value
    return {
        name: sorted((int(step), float(value)) for step, value in points.items())
        for name, points in merged.items()
    }


def final_value(series: dict[str, list[tuple[int, float]]], metric: str) -> float | None:
    points = series.get(metric, [])
    if not points:
        return None
    return points[-1][1]


def best_value(series: dict[str, list[tuple[int, float]]], metric: str) -> float | None:
    points = series.get(metric, [])
    if not points:
        return None
    return max(points, key=lambda item: item[1])[1]


def infer_notes(series: dict[str, list[tuple[int, float]]]) -> list[str]:
    notes: list[str] = []
    trunc = final_value(series, "train/truncation_rate")
    if trunc is not None and trunc > 0.5:
        notes.append("high_truncation")
    nat = final_value(series, "train/natural_termination_rate")
    if nat == 1.0:
        notes.append("natural_termination_ok")
    kl = final_value(series, "train/gen_kl_error")
    if kl is not None and kl < 1e-3:
        notes.append("kl_stable")
    gate_mean = final_value(series, "train/dg_gate_mean")
    gate_spread = final_value(series, "train/dg_gate_spread")
    if (
        gate_mean is not None
        and 0.45 <= gate_mean <= 0.55
        and (gate_spread is None or abs(gate_spread) < 0.05)
    ):
        notes.append("gate_neutral")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run specification as LABEL=RUN_DIR",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for JSON/TSV summaries")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}
    rows: list[list[Any]] = []

    for label, run_dir in args.run:
        series = load_scalars(run_dir)
        run_summary: dict[str, Any] = {
            "run_dir": run_dir,
            "notes": infer_notes(series),
        }
        for metric in SUMMARY_METRICS:
            if metric not in series:
                continue
            run_summary[metric] = {
                "final": final_value(series, metric),
                "best": best_value(series, metric),
            }
        summary[label] = run_summary
        rows.append(
            [
                label,
                run_summary.get("validation/accuracy", {}).get("final"),
                run_summary.get("train/reward", {}).get("final"),
                run_summary.get("train/truncation_rate", {}).get("final"),
                run_summary.get("train/gen_kl_error", {}).get("final"),
                run_summary.get("train/dg_gate_mean", {}).get("final"),
                run_summary.get("train/dg_gate_spread", {}).get("final"),
                ",".join(run_summary["notes"]),
            ]
        )

    with open(os.path.join(args.out_dir, "stability_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    headers = [
        "run",
        "val_acc_final",
        "reward_final",
        "trunc_final",
        "gen_kl_final",
        "gate_mean_final",
        "gate_spread_final",
        "notes",
    ]
    with open(os.path.join(args.out_dir, "stability_summary.tsv"), "w") as handle:
        handle.write("\t".join(headers) + "\n")
        for row in rows:
            handle.write("\t".join("" if x is None else str(x) for x in row) + "\n")

    widths = [max(len(headers[i]), *(len("" if row[i] is None else str(row[i])) for row in rows)) for i in range(len(headers))]
    print(" ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    for row in rows:
        print(" ".join(("" if row[i] is None else str(row[i])).ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    main()
