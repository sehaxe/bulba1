#!/usr/bin/env python3
"""
Расширенный визуализатор логов Bulba 1 (JSONL)
Улучшенная информативность: тренд loss, больше метрик в сводке.
"""

import os, sys, glob, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#adb5bd',
})


def find_log_files(logs_dir="logs"):
    patterns = [os.path.join(logs_dir, "*.jsonl"), os.path.join(logs_dir, "**", "*.jsonl")]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)


def select_log_file(files):
    if not files:
        print("No .jsonl files found")
        return None
    print("\nFound log files:")
    print("-" * 60)
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f)
        sz = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024**2:.1f} MB"
        print(f"  [{i}] {os.path.basename(f):<40} ({sz})")
    print("-" * 60)
    while True:
        try:
            choice = input(f"Select file (1-{len(files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            return None


def parse_log(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                ts = rec.get("timestamp", "")
                if " " in ts:
                    time_part = ts.split()[1]
                else:
                    time_part = ts
                data.append({
                    "time": time_part,
                    "step": int(rec["step"]),
                    "total_steps": int(rec["total_steps"]),
                    "loss": float(rec["loss"]),
                    "ema_loss": float(rec.get("ema_loss", 0)),
                    "best_loss": float(rec.get("best_loss", 0)),
                    "lr": float(rec["lr"]),
                    "stage": rec.get("stage", "unknown"),
                    "optimizer": rec.get("optimizer", "Muon+AdamW"),
                    "vram_used": int(rec["vram_used_mb"]),
                    "vram_total": int(rec["vram_total_mb"]),
                    "vram_pct": float(rec.get("vram_pct", 0)),
                    "ram_used": int(rec["ram_used_mb"]),
                    "ram_total": int(rec["ram_total_mb"]),
                    "ram_pct": float(rec.get("ram_pct", 0)),
                    "cpu_pct": int(rec["cpu_pct"]),
                    "tok_per_s": int(rec["tok_per_sec"]),
                    "oom": int(rec["oom_count"]),
                    "batch": int(rec.get("batch_size", 0)),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    if not data:
        print("No valid records found")
        return None
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} steps")
    return df


def generate_plots(df, output_path, title=None):
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        "loss": "#4263eb", "loss_trend": "#f03c3c", "ema": "#e8590c",
        "lr": "#2b8a3e", "tok": "#845ef7", "vram": "#fd7e14",
        "ram": "#fab005", "cpu": "#7950f2", "batch": "#e64980",
        "ma": "#f03c3c", "trend": "#0c8599",
    }

    # 1. Loss + EMA + Best + Trend lines
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["step"], df["loss"], color=colors["loss"], lw=1.2, alpha=0.7, label="Loss")
    ax1.plot(df["step"], df["ema_loss"], color=colors["ema"], lw=1.4, alpha=0.9, label="EMA Loss")
    ax1.axhline(df["best_loss"].min(), color=colors["loss_trend"], ls="--", lw=1.2, label="Best")
    # Trend line (linear fit)
    if len(df) > 2:
        z = np.polyfit(df["step"], df["loss"], 1)
        p = np.poly1d(z)
        ax1.plot(df["step"], p(df["step"]), "-", color=colors["trend"], lw=1.8, alpha=0.8,
                 label=f"Trend ({z[0]:.2e}/step)")
        ax1.text(0.02, 0.98, f"Slope: {z[0]:.2e}",
                 transform=ax1.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.set_title("Loss, EMA & Trend")
    ax1.legend(ncol=2, loc="upper right")

    # 2. Loss with Moving Average
    ax2 = fig.add_subplot(gs[0, 1])
    w = min(10, max(3, len(df)//10))
    df["loss_ma"] = df["loss"].rolling(window=w, center=True, min_periods=1).mean()
    ax2.plot(df["step"], df["loss"], color=colors["loss"], alpha=0.25, lw=0.8, label="Raw")
    ax2.plot(df["step"], df["loss_ma"], color=colors["ma"], lw=2, label=f"MA({w})")
    ax2.set_title("Loss with Moving Average")
    ax2.legend()

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df["step"], df["lr"], color=colors["lr"], lw=1.5)
    ax3.set_title("Learning Rate")
    ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))

    # 4. Tokens/sec
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(df["step"], df["tok_per_s"], color=colors["tok"], lw=0.8, alpha=0.4, label="tok/s")
    ax4.plot(df["step"], df["tok_per_s"].rolling(window=w, center=True, min_periods=1).mean(),
             color=colors["tok"], lw=1.8, label=f"MA({w})")
    ax4.set_ylim(df["tok_per_s"].min()*0.95, df["tok_per_s"].max()*1.02)
    ax4.set_title("Training Speed")
    ax4.legend()

    # 5. VRAM
    ax5 = fig.add_subplot(gs[1, 0])
    vram_used_gb = df["vram_used"] / 1024
    vram_total_gb = df["vram_total"] / 1024
    ax5.plot(df["step"], vram_used_gb, color=colors["vram"], lw=1.5)
    ax5.axhline(vram_total_gb.iloc[0], color="gray", ls="--", alpha=0.5)
    ax5.fill_between(df["step"], vram_used_gb, alpha=0.1, color=colors["vram"])
    ax5.set_ylim(vram_used_gb.min()*0.9, vram_total_gb.iloc[0]*1.05)
    ax5.set_title("VRAM (GB)")

    # 6. RAM
    ax6 = fig.add_subplot(gs[1, 1])
    ram_used_gb = df["ram_used"] / 1024
    ram_total_gb = df["ram_total"] / 1024
    ax6.plot(df["step"], ram_used_gb, color=colors["ram"], lw=1.5)
    ax6.axhline(ram_total_gb.iloc[0], color="gray", ls="--", alpha=0.5)
    ax6.fill_between(df["step"], ram_used_gb, alpha=0.1, color=colors["ram"])
    ax6.set_ylim(ram_used_gb.min()*0.9, ram_total_gb.iloc[0]*1.05)
    ax6.set_title("RAM (GB)")

    # 7. CPU + OOM
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(df["step"], df["cpu_pct"], color=colors["cpu"], lw=1.5, label="CPU")
    ax7.set_ylim(0, 105)
    ax7.set_title("CPU / OOM")
    oom_steps = df[df["oom"] > 0]["step"]
    if len(oom_steps):
        ax7.scatter(oom_steps, [90]*len(oom_steps), color="red", marker="x", s=50, label="OOM")

    # 8. Batch Size
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(df["step"], df["batch"], color=colors["batch"], lw=1.5, drawstyle="steps-post")
    ax8.set_title("Batch Size")
    ax8.set_ylim(bottom=0)

    # 9. Loss Histogram
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.hist(df["loss"], bins=min(30, len(df)//3), color=colors["loss"], alpha=0.6, edgecolor="white")
    ax9.set_title("Loss Distribution")

    # 10. Tok/s Distribution
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(df["tok_per_s"], bins=min(30, len(df)//3), color=colors["tok"], alpha=0.6, edgecolor="white")
    ax10.set_title("Tok/s Distribution")

    # 11. EMA vs Loss scatter (new)
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.scatter(df["loss"], df["ema_loss"], c=df["step"], cmap="viridis", alpha=0.5, s=10)
    ax11.plot([df["loss"].min(), df["loss"].max()], [df["loss"].min(), df["loss"].max()],
              "r--", lw=0.8, alpha=0.5)
    ax11.set_xlabel("Loss")
    ax11.set_ylabel("EMA Loss")
    ax11.set_title("EMA vs Loss")

    # 12. Summary (enriched)
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis("off")
    last = df.iloc[-1]
    progress = 100.0 * last["step"] / last["total_steps"] if last["total_steps"] > 0 else 0
    # Header
    ax12.text(0.5, 0.95, f"PHASE: {last['stage']}", transform=ax12.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="top",
              color="#1e40af", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe", edgecolor="#1e40af"))
    # Progress bar
    ax12.barh(0.80, progress/100.0, height=0.06, color=colors["loss"], alpha=0.7, transform=ax12.transAxes)
    ax12.barh(0.80, 1.0, height=0.06, color="none", edgecolor="#94a3b8", lw=1, transform=ax12.transAxes)
    ax12.text(0.5, 0.80, f"{progress:.1f}%", transform=ax12.transAxes, fontsize=10, ha="center", va="center")

    # Metrics
    loss_change = last["loss"] - df["loss"].iloc[0]
    trend_str = ""
    if len(df) > 2:
        z = np.polyfit(df["step"], df["loss"], 1)
        trend_str = f" | Trend: {z[0]:.2e}/step"
    ax12.text(0.5, 0.64, f"Loss: {last['loss']:.4f} | Best: {df['loss'].min():.4f}{trend_str}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.56, f"EMA: {last['ema_loss']:.4f} | Mean: {df['loss'].mean():.4f}\n"
              f"Change: {loss_change:+.4f} | Tokens: {df['step'].iloc[-1] - df['step'].iloc[0]}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.42, f"Speed: {df['tok_per_s'].mean():.0f} tok/s | Peak: {df['tok_per_s'].max()}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.32, f"VRAM: {last['vram_used']/1024:.1f}GB | RAM: {last['ram_used']/1024:.1f}GB\n"
              f"CPU: {last['cpu_pct']}% | Batch: {last['batch']}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top")
    ax12.text(0.5, 0.18, f"OOM: {int(df['oom'].sum())} | LR: {last['lr']:.2e}",
              transform=ax12.transAxes, fontsize=10, ha="center", va="top",
              color="#dc2626" if int(df['oom'].sum()) > 0 else "#16a34a")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Bulba 1 Training Visualizer (JSONL)")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output", "-o")
    parser.add_argument("--file", "-f")
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

    output = args.output or f"{Path(log_file).stem}_dashboard.png"
    title = f"Bulba 1 Training: {Path(log_file).stem.replace('_', ' ').title()}"
    result = generate_plots(df, output, title)
    print(f"\nSaved: {os.path.abspath(result)}")
    print(f"Steps: {len(df):,} | Loss: {df['loss'].iloc[0]:.3f} -> {df['loss'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()