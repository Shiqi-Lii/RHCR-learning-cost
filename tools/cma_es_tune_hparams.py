#!/usr/bin/env python3
import argparse
import json
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


def _build_learned_cmd(cfg: dict, run_dir: Path, k: int, seed: int, w: float, sigma: float, bias: float):
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
        "--use_learned_cost", "true",
        "--learned_cost_ckpt", str(cfg["learned_cost"]["ckpt"]),
        "--learned_cost_weight", str(float(w)),
        "--gaussian_sigma", str(float(sigma)),
        "--pred_bias", str(float(bias)),
    ]
    if "gaussian_ksize" in cfg["learned_cost"]:
        cmd.extend(["--gaussian_ksize", str(int(cfg["learned_cost"]["gaussian_ksize"]))])
    if "normalize" in cfg["learned_cost"]:
        cmd.extend(["--learned_cost_normalize", "true" if bool(cfg["learned_cost"]["normalize"]) else "false"])
    if cfg.get("learned_cost_python", ""):
        cmd.extend(["--learned_cost_python", str(cfg["learned_cost_python"])])
    if str(cfg["solver"]).upper() == "ECBS":
        cmd.extend(["--suboptimal_bound", str(float(cfg.get("suboptimal_bound", 1.5)))])
    return cmd


def _run_one(task: dict):
    cmd = task["cmd"]
    run_dir = task["run_dir"]
    repo_root = task["repo_root"]
    simulation_time = task["simulation_time"]
    with (run_dir / "run.log").open("w", encoding="utf-8") as logf:
        p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root / "RHCR"))
    completed = _parse_completed_tasks(run_dir / "tasks.txt")
    throughput = (completed / float(simulation_time)) if completed >= 0 else -1.0
    return p.returncode, completed, throughput


def parse_args():
    p = argparse.ArgumentParser(description="CMA-ES tune RHCR learned-cost hyperparameters.")
    p.add_argument("--runner_config", default=str(Path(__file__).resolve().parent / "rhcr_eval_config.yaml"))
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "exp/cmaes_hparams_runs"))
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--cma_sigma0", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_process", type=int, default=0, help="0 means use num_process in runner_config")
    p.add_argument("--weight_min", type=float, default=0.1)
    p.add_argument("--weight_max", type=float, default=4.0)
    p.add_argument("--sigma_min", type=float, default=0.5)
    p.add_argument("--sigma_max", type=float, default=3.0)
    p.add_argument("--bias_min", type=float, default=0.0)
    p.add_argument("--bias_max", type=float, default=2.0)
    p.add_argument("--weight_init", type=float, default=1.0)
    p.add_argument("--sigma_init", type=float, default=0.8)
    p.add_argument("--bias_init", type=float, default=0.05)
    return p.parse_args()


def main():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required. Install with: pip install optuna") from exc

    args = parse_args()
    cfg = _load_yaml(Path(args.runner_config).resolve())
    required = ["repo_root", "lifelong_bin", "rhcr_map", "scenario", "solver", "simulation_window",
                "planning_window", "simulation_time", "agents", "seeds", "learned_cost"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required key in runner config: {k}")
    if "ckpt" not in cfg["learned_cost"]:
        raise ValueError("Missing required key: learned_cost.ckpt")

    repo_root = Path(cfg["repo_root"]).resolve()
    cfg["lifelong_bin"] = str((repo_root / cfg["lifelong_bin"]).resolve()) if not str(cfg["lifelong_bin"]).startswith("/") else cfg["lifelong_bin"]
    cfg["rhcr_map"] = str((repo_root / cfg["rhcr_map"]).resolve()) if not str(cfg["rhcr_map"]).startswith("/") else cfg["rhcr_map"]
    cfg["learned_cost"]["ckpt"] = str((repo_root / cfg["learned_cost"]["ckpt"]).resolve()) if not str(cfg["learned_cost"]["ckpt"]).startswith("/") else cfg["learned_cost"]["ckpt"]
    if cfg.get("learned_cost_python", ""):
        pbin = cfg["learned_cost_python"]
        cfg["learned_cost_python"] = str((repo_root / pbin).resolve()) if not str(pbin).startswith("/") and "/" in str(pbin) else pbin

    agents = [int(x) for x in _as_list(cfg["agents"])]
    seeds = [int(x) for x in _as_list(cfg["seeds"])]
    if not agents or not seeds:
        raise ValueError("agents and seeds must be non-empty")

    num_process = int(args.num_process) if int(args.num_process) > 0 else int(cfg.get("num_process", 1))
    if num_process <= 0:
        raise ValueError("num_process must be >= 1")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() / f"rhcr_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runner_config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "tune_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    history_path = out_dir / "history.jsonl"
    best_path = out_dir / "best_candidate.json"
    study_db = out_dir / "optuna.db"
    storage_url = f"sqlite:///{study_db}"

    population = int(args.population)
    total_trials = int(args.population) * int(args.generations)
    simulation_time = int(cfg["simulation_time"])

    best = {
        "trial": -1,
        "score": float("-inf"),
        "avg_throughput": float("-inf"),
        "learned_cost_weight": float(args.weight_init),
        "gaussian_sigma": float(args.sigma_init),
        "pred_bias": float(args.bias_init),
    }

    def evaluate_candidate(weight: float, sigma: float, bias: float, trial_no: int):
        cand_dir = out_dir / f"trial_{trial_no:04d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for k in agents:
            for seed in seeds:
                run_dir = cand_dir / f"agents_{k}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = _build_learned_cmd(cfg, run_dir, k, seed, weight, sigma, bias)
                (run_dir / "command.sh").write_text(" ".join(cmd) + "\n", encoding="utf-8")
                tasks.append({
                    "cmd": cmd,
                    "run_dir": run_dir,
                    "repo_root": repo_root,
                    "simulation_time": simulation_time,
                })

        rows = []
        if num_process == 1:
            for t in tasks:
                rows.append(_run_one(t))
        else:
            with ThreadPoolExecutor(max_workers=num_process) as ex:
                futures = [ex.submit(_run_one, t) for t in tasks]
                for fut in as_completed(futures):
                    rows.append(fut.result())

        throughputs = [r[2] for r in rows if r[0] == 0 and r[2] >= 0]
        score = sum(throughputs) / len(throughputs) if throughputs else -1.0
        return score, rows

    init_score, _ = evaluate_candidate(float(args.weight_init), float(args.sigma_init), float(args.bias_init), 0)
    best.update({
        "trial": 0,
        "score": float(init_score),
        "avg_throughput": float(init_score),
    })
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"[init] score={init_score:.6f} w={args.weight_init:.4f} s={args.sigma_init:.4f} b={args.bias_init:.4f}")

    sampler = optuna.samplers.CmaEsSampler(
        x0={
            "learned_cost_weight": float(args.weight_init),
            "gaussian_sigma": float(args.sigma_init),
            "pred_bias": float(args.bias_init),
        },
        sigma0=float(args.cma_sigma0),
        seed=int(args.seed),
        n_startup_trials=0,
        popsize=population,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"rhcr_cmaes_{run_stamp}",
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial):
        weight = trial.suggest_float("learned_cost_weight", float(args.weight_min), float(args.weight_max))
        sigma = trial.suggest_float("gaussian_sigma", float(args.sigma_min), float(args.sigma_max))
        bias = trial.suggest_float("pred_bias", float(args.bias_min), float(args.bias_max))
        score, rows = evaluate_candidate(weight, sigma, bias, trial.number + 1)
        gen = trial.number // population
        idx = trial.number % population
        failed = sum(1 for r in rows if r[0] != 0)
        rec = {
            "generation": int(gen),
            "index": int(idx),
            "trial": int(trial.number + 1),
            "score": float(score),
            "avg_throughput": float(score),
            "failed_runs": int(failed),
            "learned_cost_weight": float(weight),
            "gaussian_sigma": float(sigma),
            "pred_bias": float(bias),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[gen {gen + 1}/{args.generations}][cand {idx + 1}/{population}] "
            f"score={score:.6f} w={weight:.4f} s={sigma:.4f} b={bias:.4f} failed={failed} "
            f"trial={trial.number + 1}/{total_trials}"
        )

        nonlocal best
        if score > best["score"]:
            best = rec
            best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        return float(score)

    study.optimize(objective, n_trials=total_trials)
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Saved best summary to: {best_path}")
    print(f"Saved optuna study DB to: {study_db}")


if __name__ == "__main__":
    main()
