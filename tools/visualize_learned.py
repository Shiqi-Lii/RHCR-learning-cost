#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_solver_times(run_log: Path):
    if not run_log.exists():
        return []
    times = []
    for line in run_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":Succeed," not in line:
            continue
        try:
            payload = line.split(":Succeed,", 1)[1]
            times.append(float(payload.split(",", 1)[0]))
        except Exception:
            pass
    return times


def aggregate(df, metric: str):
    agg = df.groupby(["config", "num_agents"])[metric].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]
    return agg


def plot_metric(agg, metric_label: str, title: str, out_pdf: Path, fig_w: float, fig_h: float, line_w: float):
    plt.figure(figsize=(fig_w, fig_h))
    for cfg in sorted(agg["config"].unique()):
        sub = agg[agg["config"] == cfg].sort_values("num_agents")
        x = sub["num_agents"].tolist()
        y = sub["mean"].tolist()
        ci = sub["ci95"].fillna(0).tolist()
        plt.plot(x, y, marker="o", linewidth=line_w, label=cfg)
        plt.fill_between(x, [yy - cc for yy, cc in zip(y, ci)], [yy + cc for yy, cc in zip(y, ci)], alpha=0.18)
    xticks = sorted(int(v) for v in agg["num_agents"].dropna().unique().tolist())
    if xticks:
        plt.xticks(xticks)
    plt.title(title)
    plt.xlabel("agent_num")
    plt.ylabel(metric_label)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved: {out_pdf}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize learned-cost RHCR experiment results.")
    parser.add_argument("--exp_root", default="/home/shiqi/masterarbeit/RHCR/exp_learned")
    parser.add_argument("--run_dir", action="append", default=[], help="Specific run directory. Can be passed multiple times.")
    parser.add_argument("--fig_w", type=float, default=5.5)
    parser.add_argument("--fig_h", type=float, default=5.0)
    parser.add_argument("--line_w", type=float, default=2.0)
    args = parser.parse_args()

    global np, pd, plt
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    exp_root = Path(args.exp_root).resolve()
    if args.run_dir:
        run_dirs = [Path(p).resolve() for p in args.run_dir]
    else:
        run_dirs = sorted(d for d in exp_root.rglob("*") if d.is_dir() and (d / "summary.csv").exists())
    if not run_dirs:
        raise FileNotFoundError(f"No result dirs with summary.csv found under {exp_root}")

    records = []
    for run_dir in run_dirs:
        df = pd.read_csv(run_dir / "summary.csv")
        if "mode" in df.columns:
            df = df[df["mode"] == "learned"].copy()
        if df.empty:
            continue
        for _, row in df.iterrows():
            if str(row.get("status", "")) != "ok":
                continue
            path = Path(str(row["run_dir"]))
            times = parse_solver_times(path / "run.log")
            records.append({
                "config": run_dir.name,
                "num_agents": int(row["num_agents"]),
                "seed": int(row["seed"]),
                "throughput_per_step": float(row["throughput_per_step"]),
                "mean_solver_time": float(np.mean(times)) if times else np.nan,
            })

    all_df = pd.DataFrame(records)
    if all_df.empty:
        raise RuntimeError("No learned rows found.")

    agg_thr = aggregate(all_df, "throughput_per_step")
    agg_time = aggregate(all_df.dropna(subset=["mean_solver_time"]), "mean_solver_time")

    print(agg_thr.sort_values(["num_agents", "config"]).to_string(index=False))
    plot_metric(
        agg_thr,
        "throughput_per_step",
        "Learned Throughput Comparison",
        exp_root / "learned_throughput_compare.pdf",
        args.fig_w,
        args.fig_h,
        args.line_w,
    )
    plot_metric(
        agg_time,
        "time (s)",
        "Learned Compute-Time Comparison (mean per planning call)",
        exp_root / "learned_time_compare.pdf",
        args.fig_w,
        args.fig_h,
        args.line_w,
    )


if __name__ == "__main__":
    main()
