#!/usr/bin/env python3
import argparse
from pathlib import Path


COLOR_MAP = {
    "learned cost sim=1, plan=5": "red",
    "baseline sim=1, plan=10": "orange",
    "baseline sim=1, plan=20": "green",
    "baseline sim=5, plan=10": "saddlebrown",
    "baseline sim=5, plan=20": "purple",
}


def parse_window_pairs(text: str):
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        sim_w, plan_w = item.split(":")
        out.append((int(sim_w), int(plan_w)))
    return out


def parse_runlog_solver_times(log_path: Path):
    if not log_path.exists():
        return []
    times = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":Succeed," not in line:
            continue
        try:
            payload = line.split(":Succeed,", 1)[1]
            times.append(float(payload.split(",", 1)[0]))
        except Exception:
            pass
    return times


def aggregate(df, metric: str):
    agg = df.groupby(["algorithm", "num_agents"])[metric].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]
    return agg


def latest_summary_dir(root: Path) -> Path:
    candidates = sorted(d for d in root.iterdir() if d.is_dir() and (d / "summary.csv").exists())
    if not candidates:
        raise FileNotFoundError(f"No result dirs with summary.csv found under {root}")
    return candidates[-1]


def plot_lines(agg, order, metric_label: str, title: str, out_pdf: Path, fig_w: float, fig_h: float, line_w: float):
    plt.figure(figsize=(fig_w, fig_h))
    for algo in order:
        sub = agg[agg["algorithm"] == algo].sort_values("num_agents")
        if sub.empty:
            continue
        x = sub["num_agents"].tolist()
        y = sub["mean"].tolist()
        ci = sub["ci95"].fillna(0).tolist()
        color = COLOR_MAP.get(algo)
        plt.plot(x, y, marker="o", linewidth=line_w, label=algo, color=color)
        plt.fill_between(x, [yy - cc for yy, cc in zip(y, ci)], [yy + cc for yy, cc in zip(y, ci)], alpha=0.18, color=color)
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
    parser = argparse.ArgumentParser(description="Visualize RHCR baseline window sweep vs learned-cost results.")
    parser.add_argument("--diff_root", default="/home/shiqi/masterarbeit/RHCR/exp_diff_sim_plan")
    parser.add_argument("--learned_root", default="/home/shiqi/masterarbeit/RHCR/exp_learned/rhcr_eval_runs")
    parser.add_argument("--diff_dir", "--exp2_dir", dest="diff_dir", default="", help="Specific diff sim/plan result dir.")
    parser.add_argument("--learned_dir", "--cmp_dir", dest="learned_dir", default="", help="Specific learned-cost result dir.")
    parser.add_argument("--window_pairs", default="1:5,1:10,1:20,5:10,5:20")
    parser.add_argument("--fig_w", type=float, default=6.0)
    parser.add_argument("--fig_h", type=float, default=5.0)
    parser.add_argument("--line_w", type=float, default=2.0)
    args = parser.parse_args()

    global np, pd, plt
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    diff_dir = Path(args.diff_dir).resolve() if args.diff_dir else latest_summary_dir(Path(args.diff_root).resolve())
    cmp_dir = Path(args.learned_dir).resolve() if args.learned_dir else latest_summary_dir(Path(args.learned_root).resolve())
    diff_csv = diff_dir / "summary.csv"
    cmp_csv = cmp_dir / "summary.csv"
    if not diff_csv.exists():
        raise FileNotFoundError(diff_csv)
    if not cmp_csv.exists():
        raise FileNotFoundError(cmp_csv)

    window_pairs = parse_window_pairs(args.window_pairs)
    order = [f"baseline sim={s}, plan={p}" for s, p in window_pairs] + ["learned cost sim=1, plan=5"]

    diff_df = pd.read_csv(diff_csv)
    cmp = pd.read_csv(cmp_csv)
    for df in (diff_df, cmp):
        df["num_agents"] = pd.to_numeric(df["num_agents"], errors="coerce").astype("Int64")
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
        df["throughput_per_step"] = pd.to_numeric(df["throughput_per_step"], errors="coerce")

    parts = []
    for sim_w, plan_w in window_pairs:
        sub = diff_df[
            (diff_df["status"] == "ok")
            & (diff_df["simulation_window"] == sim_w)
            & (diff_df["planning_window"] == plan_w)
        ].copy()
        sub["algorithm"] = f"baseline sim={sim_w}, plan={plan_w}"
        parts.append(sub[["algorithm", "num_agents", "seed", "throughput_per_step", "run_dir"]])
    learned = cmp[(cmp["status"] == "ok") & (cmp["mode"] == "learned")].copy()
    learned["algorithm"] = "learned cost sim=1, plan=5"
    parts.append(learned[["algorithm", "num_agents", "seed", "throughput_per_step", "run_dir"]])
    df = pd.concat(parts, ignore_index=True)

    agg_thr = aggregate(df, "throughput_per_step")
    print(agg_thr.sort_values(["num_agents", "algorithm"]).to_string(index=False))
    plot_lines(
        agg_thr,
        order,
        "throughput_per_step",
        "RHCR Combined: learned + baseline(sim/plan variants)",
        cmp_dir / "combined_diff_sim_plan_vs_learned.pdf",
        args.fig_w,
        args.fig_h,
        args.line_w,
    )

    runtime_rows = []
    for _, row in df.iterrows():
        times = parse_runlog_solver_times(Path(str(row["run_dir"])) / "run.log")
        if not times:
            continue
        runtime_rows.append({
            "algorithm": row["algorithm"],
            "num_agents": int(row["num_agents"]),
            "seed": int(row["seed"]),
            "mean_solver_time": float(np.mean(times)),
        })
    rt = pd.DataFrame(runtime_rows)
    if rt.empty:
        raise RuntimeError("No runtime data parsed from run.log")
    agg_time = aggregate(rt, "mean_solver_time")
    plot_lines(
        agg_time,
        order,
        "time (s)",
        "Compute Time Comparison (mean per planning call)",
        cmp_dir / "combined_compute_time_all_curves.pdf",
        args.fig_w,
        args.fig_h,
        args.line_w,
    )


if __name__ == "__main__":
    main()
