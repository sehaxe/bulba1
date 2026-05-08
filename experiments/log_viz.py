#!/usr/bin/env python3
"""
Training Log Visualizer
Generates beautiful training graphs from log files
"""

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def find_log_files(logs_dir="logs"):
    """Find all .log files in logs directory"""
    patterns = [
        os.path.join(logs_dir, "*.log"),
        os.path.join(logs_dir, "**", "*.log"),
    ]
    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    return sorted(files)


def select_log_file(files):
    """Interactive file selection"""
    if not files:
        print("No .log files found in logs/ directory")
        return None

    print("\nFound log files:")
    print("-" * 60)
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f)
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
        print(f"  [{i}] {os.path.basename(f):<40} ({size_str})")
    print("-" * 60)

    while True:
        try:
            choice = input(f"\nSelect file number (1-{len(files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            print("Invalid number")
        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            print("\nCancelled")
            return None


def parse_log(filepath):
    """Parse log file, filtering garbage"""

    # Regex for training log line
    pattern = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\]\s+"  # [HH:MM:SS]
        r"Step\s+(\d+)/(\d+)\s+\|\s+"  # Step X/Y |
        r"loss=([\d.]+)\s+\|\s+"  # loss=... |
        r"(\w+)\s+\|\s+"  # Phase |
        r"(\w+)\s+\|\s+"  # Optimizer |
        r"LR=([\d.e+-]+)\s+\|\s+"  # LR=... |
        r"VRAM=(\d+)/(\d+)MB\((\d+)%\)\s+"  # VRAM=... |
        r"RAM=(\d+)/(\d+)MB\((\d+)%\)\s+"  # RAM=... |
        r"CPU=(\d+)%\s+\|\s+"  # CPU=... |
        r"tok/s=(\d+)\s+"  # tok/s=...
        r"OOM=(\d+)"  # OOM=...
    )

    data = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                data.append(
                    {
                        "time": match.group(1),
                        "step": int(match.group(2)),
                        "total_steps": int(match.group(3)),
                        "loss": float(match.group(4)),
                        "phase": match.group(5),
                        "optimizer": match.group(6),
                        "lr": float(match.group(7)),
                        "vram_used": int(match.group(8)),
                        "vram_total": int(match.group(9)),
                        "vram_pct": int(match.group(10)),
                        "ram_used": int(match.group(11)),
                        "ram_total": int(match.group(12)),
                        "ram_pct": int(match.group(13)),
                        "cpu_pct": int(match.group(14)),
                        "tok_per_s": int(match.group(15)),
                        "oom": int(match.group(16)),
                    }
                )
            else:
                skipped += 1

    if not data:
        print(f"Failed to parse any lines from {filepath}")
        return None

    df = pd.DataFrame(data)
    print(f"Parsed: {len(df)} lines, skipped (garbage): {skipped}")
    return df


def generate_plots(df, output_path, title=None):
    """Generate beautiful plots"""

    vram_real = df["vram_used"].max()

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Color scheme
    colors = {
        "loss": "#2563eb",
        "loss_trend": "#dc2626",
        "lr": "#16a34a",
        "tok": "#9333ea",
        "vram": "#ea580c",
        "ram": "#ca8a04",
        "cpu": "#7c2d12",
        "ma": "#dc2626",
    }

    # === 1. Loss over Steps (with trend) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["step"], df["loss"], color=colors["loss"], linewidth=1.2, alpha=0.7, label="Loss")

    if len(df) > 2:
        z = np.polyfit(df["step"], df["loss"], 1)
        p = np.poly1d(z)
        ax1.plot(
            df["step"],
            p(df["step"]),
            "--",
            color=colors["loss_trend"],
            linewidth=1.5,
            alpha=0.8,
            label=f"Trend: {z[0]:.2e}",
        )

    ax1.set_xlabel("Step", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Loss over Steps", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)

    # === 2. Loss with Moving Average ===
    ax2 = fig.add_subplot(gs[0, 1])
    window = min(10, max(3, len(df) // 10))
    df["loss_ma"] = df["loss"].rolling(window=window, center=True, min_periods=1).mean()

    ax2.plot(df["step"], df["loss"], color=colors["loss"], alpha=0.25, linewidth=0.8, label="Raw")
    ax2.plot(df["step"], df["loss_ma"], color=colors["ma"], linewidth=2, label=f"MA({window})")
    ax2.set_xlabel("Step", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.set_title("Loss with Moving Average", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8)

    # === 3. Learning Rate ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df["step"], df["lr"], color=colors["lr"], linewidth=1.5)
    ax3.set_xlabel("Step", fontsize=10)
    ax3.set_ylabel("Learning Rate", fontsize=10)
    ax3.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.2)
    ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # === 4. Training Speed ===
    ax4 = fig.add_subplot(gs[1, 0])
    window = min(10, max(3, len(df) // 10))
    df["tok_ma"] = df["tok_per_s"].rolling(window=window, center=True, min_periods=1).mean()
    ax4.plot(
        df["step"], df["tok_per_s"], color=colors["tok"], linewidth=0.8, alpha=0.4, label="Raw"
    )
    ax4.plot(df["step"], df["tok_ma"], color=colors["tok"], linewidth=1.8, label=f"MA({window})")
    ax4.fill_between(df["step"], df["tok_ma"], alpha=0.1, color=colors["tok"])
    y_min = df["tok_per_s"].min() * 0.95
    y_max = df["tok_per_s"].max() * 1.02
    ax4.set_ylim(y_min, y_max)
    ax4.set_xlabel("Step", fontsize=10)
    ax4.set_ylabel("tokens/sec", fontsize=10)
    ax4.set_title("Training Speed", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=8)

    # === 5. VRAM ===
    ax5 = fig.add_subplot(gs[1, 1])
    vram_used_gb = df["vram_used"] / 1024
    vram_total_gb = df["vram_total"] / 1024

    y_min = vram_used_gb.min() * 0.9
    y_max = vram_total_gb.iloc[0] * 1.05
    ax5.plot(
        df["step"],
        vram_used_gb,
        color=colors["vram"],
        linewidth=1.5,
        label=f"Used: {vram_used_gb.iloc[-1]:.1f}GB",
    )
    ax5.axhline(
        y=vram_total_gb.iloc[0],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Total: {vram_total_gb.iloc[0]:.1f}GB",
    )
    ax5.axhline(y=vram_total_gb.iloc[0] * 0.8, color="red", linestyle="--", alpha=0.5, label="80%")
    ax5.fill_between(df["step"], vram_used_gb, alpha=0.1, color=colors["vram"])

    ax5.set_ylim(y_min, y_max)
    ax5.set_title("VRAM", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=8, loc="upper left")
    ax5.set_xlabel("Step", fontsize=10)
    ax5.set_ylabel("GB", fontsize=10)
    ax5.grid(True, alpha=0.2)

    ax6 = fig.add_subplot(gs[1, 2])
    ram_used_gb = df["ram_used"] / 1024
    ram_total_gb = df["ram_total"] / 1024

    y_min = ram_used_gb.min() * 0.9
    y_max = ram_total_gb.iloc[0] * 1.05
    ax6.plot(
        df["step"],
        ram_used_gb,
        color=colors["ram"],
        linewidth=1.5,
        label=f"Used: {ram_used_gb.iloc[-1]:.1f}GB",
    )
    ax6.axhline(
        y=ram_total_gb.iloc[0],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Total: {ram_total_gb.iloc[0]:.1f}GB",
    )
    ax6.axhline(y=ram_total_gb.iloc[0] * 0.8, color="red", linestyle="--", alpha=0.5, label="80%")
    ax6.fill_between(df["step"], ram_used_gb, alpha=0.1, color=colors["ram"])

    ax6.set_ylim(y_min, y_max)
    ax6.set_xlabel("Step", fontsize=10)
    ax6.set_ylabel("GB", fontsize=10)
    ax6.set_title("RAM", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.2)
    ax6.legend(fontsize=8, loc="upper left")

    # === 7. CPU Usage ===
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(df["step"], df["cpu_pct"], color=colors["cpu"], linewidth=1.5)
    ax7.fill_between(df["step"], df["cpu_pct"], alpha=0.1, color=colors["cpu"])
    ax7.set_xlabel("Step", fontsize=10)
    ax7.set_ylabel("CPU %", fontsize=10)
    ax7.set_title("CPU Usage (%)", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.2)
    ax7.set_ylim(0, 105)

    # === 8. Loss Distribution (Histogram) ===
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(
        df["loss"], bins=min(30, len(df) // 3), color=colors["loss"], alpha=0.6, edgecolor="white"
    )
    ax8.axvline(
        df["loss"].mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean: {df['loss'].mean():.3f}",
    )
    ax8.axvline(
        df["loss"].median(),
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Median: {df['loss'].median():.3f}",
    )
    ax8.set_xlabel("Loss", fontsize=10)
    ax8.set_ylabel("Frequency", fontsize=10)
    ax8.set_title("Loss Distribution", fontsize=12, fontweight="bold")
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.2, axis="y")

    # === 9. Summary Statistics ===
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    loss_change = df["loss"].iloc[-1] - df["loss"].iloc[0]
    if len(df) > 2:
        slope, _ = np.polyfit(df["step"], df["loss"], 1)
    else:
        slope = 0

    vram_gb = df["vram_used"].iloc[-1] / 1024
    vram_total_gb = df["vram_total"].iloc[0] / 1024
    ram_gb = df["ram_used"].iloc[-1] / 1024
    ram_total_gb = df["ram_total"].iloc[0] / 1024

    phase = df["phase"].iloc[-1]
    optimizer = df["optimizer"].iloc[-1]
    lr = df["lr"].iloc[-1]
    progress = 100 * df["step"].iloc[-1] / df["total_steps"].iloc[0]

    ax9.text(
        0.5,
        0.95,
        f"PHASE: {phase}",
        transform=ax9.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        color="#1e40af",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe", edgecolor="#1e40af", linewidth=2),
    )

    progress = min(progress / 100, 1.0)
    ax9.barh(0.80, progress, height=0.06, color=colors["loss"], alpha=0.7, transform=ax9.transAxes)
    ax9.barh(
        0.80,
        1.0,
        height=0.06,
        color="none",
        edgecolor="#94a3b8",
        linewidth=1,
        transform=ax9.transAxes,
    )
    ax9.text(
        0.5,
        0.80,
        f"{100 * df['step'].iloc[-1] / df['total_steps'].iloc[0]:.1f}%",
        transform=ax9.transAxes,
        fontsize=10,
        ha="center",
        va="center",
    )

    ax9.text(
        0.5,
        0.62,
        f"Loss: {df['loss'].iloc[-1]:.4f} | Best: {df['loss'].min():.4f} | Mean: {df['loss'].mean():.4f}",
        transform=ax9.transAxes,
        fontsize=10,
        ha="center",
        va="top",
    )
    ax9.text(
        0.5,
        0.50,
        f"Speed: {df['tok_per_s'].mean():.0f} tok/s | Peak: {df['tok_per_s'].max()}",
        transform=ax9.transAxes,
        fontsize=10,
        ha="center",
        va="top",
    )

    oom_count = df["oom"].sum()
    oom_color = "#dc2626" if oom_count > 0 else "#16a34a"
    ax9.text(
        0.5,
        0.40,
        f"OOMs: {oom_count}",
        transform=ax9.transAxes,
        fontsize=10,
        ha="center",
        va="top",
        color=oom_color,
        fontweight="bold",
    )

    ax9.text(
        0.5,
        0.28,
        f"VRAM: {vram_gb:.1f}GB | RAM: {ram_gb:.1f}GB | CPU: {df['cpu_pct'].iloc[-1]}%",
        transform=ax9.transAxes,
        fontsize=10,
        ha="center",
        va="top",
    )
    ax9.text(
        0.5,
        0.12,
        f"Optimizer: {optimizer} | LR: {lr:.2e}",
        transform=ax9.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        color="#64748b",
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--logs-dir", default="logs", help="Directory with log files")
    parser.add_argument("--output", "-o", help="Output image path (auto if not set)")
    parser.add_argument("--file", "-f", help="Specific log file (skip selection)")
    args = parser.parse_args()

    if args.file:
        log_file = args.file
        if not os.path.exists(log_file):
            print(f"File not found: {log_file}")
            return
    else:
        files = find_log_files(args.logs_dir)
        log_file = select_log_file(files)
        if not log_file:
            return

    print(f"\nSelected: {log_file}")

    df = parse_log(log_file)
    if df is None:
        return

    if args.output:
        output = args.output
    else:
        base = Path(log_file).stem
        output = f"{base}_viz.png"

    title = f"Training: {Path(log_file).stem.replace('_', ' ').title()}"
    result = generate_plots(df, output, title)

    print(f"\nSaved: {os.path.abspath(result)}")
    print(f"Steps: {len(df):,} | Loss: {df['loss'].iloc[0]:.3f} -> {df['loss'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
