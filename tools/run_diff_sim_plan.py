#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_yaml(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_completed_tasks(tasks_file: Path) -> int:
    if not tasks_file.exists():
        return -1
    total = 0
    for line in tasks_file.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        for segment in [s for s in line.split(";") if s.strip()]:
            parts = [p for p in segment.split(",") if p != ""]
            if len(parts) != 3:
                continue
            try:
                finish_time = int(parts[1])
            except ValueError:
                continue
            if finish_time >= 0:
                total += 1
    return total


def _parse_solver_times(log_file: Path) -> list[float]:
    if not log_file.exists():
        return []
    times = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":Succeed," not in line:
            continue
        try:
            payload = line.split(":Succeed,", 1)[1]
            times.append(float(payload.split(",", 1)[0]))
        except (IndexError, ValueError):
            continue
    return times


def _read_log_tail(log_file: Path, max_lines: int = 80) -> list[str]:
    if not log_file.exists():
        return []
    lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def _resolve_path(repo_root: Path, value: str) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (repo_root / path).resolve())


def _build_cmd(cfg: dict, run_dir: Path, sim_w: int, plan_w: int, k: int, seed: int):
    cmd = [
        str(Path(cfg["lifelong_bin"]).resolve()),
        "-m", str(cfg["rhcr_map"]),
        "--scenario", str(cfg["scenario"]),
        "-k", str(k),
        "--simulation_window", str(sim_w),
        "--planning_window", str(plan_w),
        "--solver", str(cfg["solver"]),
        "--seed", str(seed),
        "--simulation_time", str(int(cfg["simulation_time"])),
        "--dummy_paths", "true" if bool(cfg.get("dummy_paths", False)) else "false",
        "--use_learned_cost", "false",
        "-o", str(run_dir),
    ]
    if str(cfg["solver"]).upper() == "ECBS":
        cmd.extend(["--suboptimal_bound", str(float(cfg.get("suboptimal_bound", 1.5)))])
    return cmd


def _run_one(task: dict) -> list:
    sim_w = task["sim_w"]
    plan_w = task["plan_w"]
    k = task["k"]
    seed = task["seed"]
    run_dir = task["run_dir"]
    simulation_time = task["simulation_time"]

    status = "ok"
    returncode = 0
    if task["dry_run"]:
        status = "dry_run"
    else:
        with (run_dir / "run.log").open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                task["cmd"],
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(task["rhcr_root"]),
            )
        returncode = proc.returncode
        if returncode != 0:
            status = "failed"

    completed = _parse_completed_tasks(run_dir / "tasks.txt")
    solver_times = _parse_solver_times(run_dir / "run.log")
    planning_calls = len(solver_times)
    total_solver_time = sum(solver_times) if solver_times else -1.0
    mean_solver_time = (total_solver_time / planning_calls) if planning_calls else -1.0
    throughput = (completed / float(simulation_time)) if completed >= 0 else -1.0
    raw = {
        "simulation_window": sim_w,
        "planning_window": plan_w,
        "num_agents": k,
        "seed": seed,
        "status": status,
        "returncode": returncode,
        "completed_tasks": completed,
        "throughput_per_step": throughput,
        "mean_solver_time": mean_solver_time,
        "total_solver_time": total_solver_time,
        "planning_calls": planning_calls,
        "solver_times": solver_times,
    }
    if status != "ok":
        raw["log_tail"] = _read_log_tail(run_dir / "run.log")
    if task["cleanup_run_dir"] and run_dir.exists():
        shutil.rmtree(run_dir)
    print(
        f"[{status}] sim={sim_w} plan={plan_w} k={k} seed={seed} "
        f"completed={completed} throughput={throughput:.4f}"
    )
    row = [
        sim_w, plan_w, k, seed, status, returncode, completed, throughput,
        mean_solver_time, total_solver_time, planning_calls,
    ]
    return row, raw


def main():
    parser = argparse.ArgumentParser(description="Run RHCR baseline over simulation/planning window combinations.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "run_diff_sim_plan.yaml"),
        help="Path to yaml config",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config).resolve())
    required = [
        "repo_root", "lifelong_bin", "rhcr_map", "scenario", "solver",
        "simulation_time", "agents", "seeds", "window_pairs", "output_root",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required key in config: {key}")

    repo_root = Path(cfg["repo_root"]).resolve()
    rhcr_root = repo_root / "RHCR"
    cfg["lifelong_bin"] = _resolve_path(repo_root, cfg["lifelong_bin"])
    cfg["rhcr_map"] = _resolve_path(repo_root, cfg["rhcr_map"])

    lifelong_bin = Path(cfg["lifelong_bin"])
    map_path = Path(cfg["rhcr_map"])
    if not lifelong_bin.exists():
        raise FileNotFoundError(f"Missing binary: {lifelong_bin}")
    if not map_path.exists():
        raise FileNotFoundError(f"Missing map: {map_path}")

    agents = [int(x) for x in _as_list(cfg["agents"])]
    seeds = [int(x) for x in _as_list(cfg["seeds"])]
    window_pairs = [(int(x[0]), int(x[1])) for x in cfg["window_pairs"]]
    num_process = int(cfg.get("num_process", 1))
    if num_process <= 0:
        raise ValueError("num_process must be >= 1")

    output_root = Path(_resolve_path(repo_root, cfg["output_root"]))
    map_stem = Path(cfg["rhcr_map"]).stem
    run_name = cfg.get("run_name") or f"{map_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    temp_root = run_root / "_tmp"
    temp_root.mkdir(exist_ok=True)
    cleanup_run_dir = bool(cfg.get("cleanup_run_dir", True))

    summary_path = run_root / "summary.csv"
    raw_path = run_root / "runs_raw.jsonl"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "simulation_window", "planning_window", "num_agents", "seed",
            "status", "returncode", "completed_tasks", "throughput_per_step",
            "mean_solver_time", "total_solver_time", "planning_calls",
        ])

    tasks = []
    for sim_w, plan_w in window_pairs:
        for k in agents:
            for seed in seeds:
                run_dir = temp_root / f"sim_{sim_w}_plan_{plan_w}_agents_{k}_seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = _build_cmd(cfg, run_dir, sim_w, plan_w, k, seed)
                tasks.append({
                    "sim_w": sim_w,
                    "plan_w": plan_w,
                    "k": k,
                    "seed": seed,
                    "cmd": cmd,
                    "run_dir": run_dir,
                    "rhcr_root": rhcr_root,
                    "simulation_time": int(cfg["simulation_time"]),
                    "dry_run": bool(cfg.get("dry_run", False)),
                    "cleanup_run_dir": cleanup_run_dir,
                })

    print(f"run_root = {run_root}")
    print(f"Launching {len(tasks)} runs with num_process={num_process}")

    rows = []
    if num_process == 1:
        for task in tasks:
            rows.append(_run_one(task))
    else:
        with ThreadPoolExecutor(max_workers=num_process) as executor:
            futures = [executor.submit(_run_one, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda item: (int(item[0][0]), int(item[0][1]), int(item[0][2]), int(item[0][3])))
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(row for row, _ in rows)
    with raw_path.open("w", encoding="utf-8") as f:
        for _, raw in rows:
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
    if cleanup_run_dir and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
