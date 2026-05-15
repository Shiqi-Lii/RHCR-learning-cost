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


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _parse_completed_tasks(tasks_file: Path) -> int:
    if not tasks_file.exists():
        return -1
    lines = tasks_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    total = 0
    for ln in lines[1:]:
        for seg in [s for s in ln.split(";") if s.strip()]:
            parts = [p for p in seg.split(",") if p != ""]
            if len(parts) == 3:
                try:
                    t = int(parts[1])
                except ValueError:
                    continue
                if t >= 0:
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


def _build_base_cmd(cfg: dict, run_dir: Path, k: int, seed: int):
    cmd = [
        str(Path(cfg["lifelong_bin"]).resolve()),
        "-m", str(cfg["rhcr_map"]),
        "--scenario", str(cfg["scenario"]),
        "-k", str(k),
        "--simulation_window", str(int(cfg["simulation_window"])),
        "--planning_window", str(int(cfg["planning_window"])),
        "--solver", str(cfg["solver"]),
        "--seed", str(seed),
        "--simulation_time", str(int(cfg["simulation_time"])),
        "--dummy_paths", "true" if bool(cfg.get("dummy_paths", False)) else "false",
        "-o", str(run_dir),
    ]
    if str(cfg["solver"]).upper() == "ECBS":
        cmd.extend(["--suboptimal_bound", str(float(cfg.get("suboptimal_bound", 1.5)))])
    return cmd


def _build_learned_cmd(cfg: dict, run_dir: Path, k: int, seed: int):
    lc = cfg.get("learned_cost", {})
    cmd = _build_base_cmd(cfg, run_dir, k, seed)
    cmd.extend(["--use_learned_cost", "true"])
    cmd.extend(["--learned_cost_ckpt", str(lc["ckpt"])])
    if "weight" in lc:
        cmd.extend(["--learned_cost_weight", str(float(lc["weight"]))])
    if "gaussian_sigma" in lc:
        cmd.extend(["--gaussian_sigma", str(float(lc["gaussian_sigma"]))])
    if "pred_bias" in lc:
        cmd.extend(["--pred_bias", str(float(lc["pred_bias"]))])
    if "gaussian_ksize" in lc:
        cmd.extend(["--gaussian_ksize", str(int(lc["gaussian_ksize"]))])
    if "normalize" in lc:
        cmd.extend(["--learned_cost_normalize", "true" if bool(lc["normalize"]) else "false"])
    return cmd


def _run_one(task: dict) -> list:
    mode = task["mode"]
    k = task["k"]
    seed = task["seed"]
    cmd = task["cmd"]
    run_dir = task["run_dir"]
    repo_root = task["repo_root"]
    simulation_time = task["simulation_time"]
    dry_run = task["dry_run"]

    status = "ok"
    returncode = 0
    if dry_run:
        status = "dry_run"
    else:
        with (run_dir / "run.log").open("w", encoding="utf-8") as logf:
            p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root / "RHCR"))
        returncode = p.returncode
        if returncode != 0:
            status = "failed"

    completed = _parse_completed_tasks(run_dir / "tasks.txt")
    solver_times = _parse_solver_times(run_dir / "run.log")
    planning_calls = len(solver_times)
    total_solver_time = sum(solver_times) if solver_times else -1.0
    mean_solver_time = (total_solver_time / planning_calls) if planning_calls else -1.0
    throughput = (completed / float(simulation_time)) if completed >= 0 else -1.0
    raw = {
        "mode": mode,
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
    print(f"[{status}] mode={mode} k={k} seed={seed} completed={completed} throughput={throughput:.4f}")
    row = [
        mode, k, seed, status, returncode, completed, throughput,
        mean_solver_time, total_solver_time, planning_calls,
    ]
    return row, raw


def main():
    parser = argparse.ArgumentParser(description="RHCR learned-cost runner over agents x seeds.")
    parser.add_argument(
        "--runner_config",
        default=str(Path(__file__).resolve().parent / "run_learnedcost_exp.yaml"),
        help="Path to RHCR runner yaml config",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.runner_config).resolve())
    required = ["repo_root", "lifelong_bin", "rhcr_map", "scenario", "solver", "simulation_window",
                "planning_window", "simulation_time", "agents", "seeds", "output_root", "learned_cost"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required key in runner config: {k}")
    if "ckpt" not in cfg["learned_cost"]:
        raise ValueError("Missing required key: learned_cost.ckpt")

    repo_root = Path(cfg["repo_root"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    agents = [int(x) for x in _as_list(cfg["agents"])]
    seeds = [int(x) for x in _as_list(cfg["seeds"])]
    if not agents or not seeds:
        raise ValueError("agents and seeds must be non-empty")
    num_process = int(cfg.get("num_process", 1))
    if num_process <= 0:
        raise ValueError("num_process must be >= 1")

    # Resolve paths
    cfg["lifelong_bin"] = str((repo_root / cfg["lifelong_bin"]).resolve()) if not str(cfg["lifelong_bin"]).startswith("/") else cfg["lifelong_bin"]
    cfg["rhcr_map"] = str((repo_root / cfg["rhcr_map"]).resolve()) if not str(cfg["rhcr_map"]).startswith("/") else cfg["rhcr_map"]
    cfg["learned_cost"]["ckpt"] = str((repo_root / cfg["learned_cost"]["ckpt"]).resolve()) if not str(cfg["learned_cost"]["ckpt"]).startswith("/") else cfg["learned_cost"]["ckpt"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = output_root / f"rhcr_compare_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    temp_root = out_root / "_tmp"
    temp_root.mkdir(exist_ok=True)
    cleanup_run_dir = bool(cfg.get("cleanup_run_dir", True))

    summary_path = out_root / "summary.csv"
    raw_path = out_root / "runs_raw.jsonl"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "num_agents", "seed", "status", "returncode", "completed_tasks",
            "throughput_per_step", "mean_solver_time", "total_solver_time", "planning_calls",
        ])

    tasks = []
    mode = "learned"
    for k in agents:
        for seed in seeds:
            run_dir = temp_root / f"{mode}_agents_{k}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = _build_learned_cmd(cfg, run_dir, k, seed)
            tasks.append({
                "mode": mode,
                "k": k,
                "seed": seed,
                "cmd": cmd,
                "run_dir": run_dir,
                "repo_root": repo_root,
                "simulation_time": cfg["simulation_time"],
                "dry_run": bool(cfg.get("dry_run", False)),
                "cleanup_run_dir": cleanup_run_dir,
            })

    print(f"Launching {len(tasks)} runs with num_process={num_process}")
    rows = []
    if num_process == 1:
        for task in tasks:
            rows.append(_run_one(task))
    else:
        with ThreadPoolExecutor(max_workers=num_process) as ex:
            futures = [ex.submit(_run_one, task) for task in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())

    rows.sort(key=lambda item: (item[0][0], int(item[0][1]), int(item[0][2])))
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row, _ in rows:
            w.writerow(row)
    with raw_path.open("w", encoding="utf-8") as f:
        for _, raw in rows:
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
    if cleanup_run_dir and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
