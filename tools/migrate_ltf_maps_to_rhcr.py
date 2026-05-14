#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
from collections import deque
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect_target_maps(experiments_root: Path) -> dict[str, int]:
    """Return map_name -> max_num_agents from experiments_cmaes yaml files."""
    out: dict[str, int] = {}
    for yml in experiments_root.rglob("*.yaml"):
        cfg = load_yaml(yml)
        env = cfg.get("environment", {})
        map_name = env.get("map_name")
        if not map_name:
            continue
        if map_name == "wfi_warehouse":
            continue
        num_cfg = env.get("num_agents", {})
        grid = num_cfg.get("grid_search", [])
        if not grid:
            continue
        m = int(max(int(x) for x in grid))
        out[map_name] = max(out.get(map_name, 0), m)
    return out


def load_ltf_maps(env_yaml_paths: list[Path]) -> dict[str, list[str]]:
    maps: dict[str, list[str]] = {}
    for p in env_yaml_paths:
        cfg = load_yaml(p)
        for name, ascii_block in cfg.items():
            if not isinstance(ascii_block, str):
                continue
            rows = [ln.rstrip() for ln in ascii_block.splitlines() if ln.strip() != ""]
            if rows:
                maps[name] = rows
    return maps


def _degree(idx: int, rows: int, cols: int, is_free: list[bool]) -> int:
    r = idx // cols
    c = idx % cols
    d = 0
    if c + 1 < cols and is_free[idx + 1]:
        d += 1
    if c - 1 >= 0 and is_free[idx - 1]:
        d += 1
    if r + 1 < rows and is_free[idx + cols]:
        d += 1
    if r - 1 >= 0 and is_free[idx - cols]:
        d += 1
    return d


def choose_homes(free_cells: list[int], home_count: int, map_name: str, rows: int, cols: int) -> set[int]:
    if home_count > len(free_cells):
        raise ValueError(f"{map_name}: home_count={home_count} exceeds free cells={len(free_cells)}")
    is_free = [False] * (rows * cols)
    for i in free_cells:
        is_free[i] = True
    # Prefer high-connectivity cells for homes to avoid dead-ends and fragile starts.
    candidates = [i for i in free_cells if _degree(i, rows, cols, is_free) >= 3]
    if len(candidates) < home_count:
        candidates = [i for i in free_cells if _degree(i, rows, cols, is_free) >= 2]
    if len(candidates) < home_count:
        candidates = free_cells
    seed = int(hashlib.sha256(map_name.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return set(rng.sample(candidates, home_count))


def build_rhcr_map(grid: list[str], map_name: str, home_count: int) -> tuple[list[str], int, int]:
    rows = len(grid)
    cols = len(grid[0])
    for r in grid:
        if len(r) != cols:
            raise ValueError(f"{map_name}: inconsistent row length")

    free = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != "#":
                free.append(i * cols + j)

    # Keep only the largest connected free-space component to avoid unreachable goals in KIVA.
    is_free0 = [False] * (rows * cols)
    for idx in free:
        is_free0[idx] = True

    visited = [False] * (rows * cols)
    comps: list[list[int]] = []
    for s in free:
        if visited[s]:
            continue
        q = deque([s])
        visited[s] = True
        comp = [s]
        while q:
            u = q.popleft()
            r = u // cols
            c = u % cols
            nbrs = []
            if c + 1 < cols:
                nbrs.append(u + 1)
            if c - 1 >= 0:
                nbrs.append(u - 1)
            if r + 1 < rows:
                nbrs.append(u + cols)
            if r - 1 >= 0:
                nbrs.append(u - cols)
            for v in nbrs:
                if is_free0[v] and not visited[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        comps.append(comp)

    largest = max(comps, key=len) if comps else []
    free_main = sorted(largest)
    if len(free_main) < home_count:
        raise ValueError(
            f"{map_name}: largest connected component too small "
            f"({len(free_main)}) for home_count={home_count}"
        )

    free_main_set = set(free_main)
    homes = choose_homes(free_main, home_count, map_name, rows, cols)
    endpoints = free_main_set - homes

    out_rows: list[str] = []
    for i in range(rows):
        chars = []
        for j in range(cols):
            idx = i * cols + j
            if idx not in free_main_set:
                chars.append("@")
            elif idx in homes:
                chars.append("r")
            elif idx in endpoints:
                chars.append("e")
            else:
                chars.append(".")
        out_rows.append("".join(chars))

    return out_rows, len(endpoints), len(homes)


def main():
    ap = argparse.ArgumentParser(description="Migrate learn-to-follow maps into RHCR KIVA .map format.")
    ap.add_argument("--repo_root", default="/home/shiqi/masterarbeit")
    ap.add_argument("--max_time", type=int, default=5000)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ltf_root = repo_root / "learn-to-follow"
    rhcr_maps = repo_root / "RHCR" / "maps"
    exp_root = ltf_root / "experiments_cmaes"
    env_yamls = [ltf_root / "env" / "test-maps.yaml", ltf_root / "env" / "training-maps.yaml"]

    targets = collect_target_maps(exp_root)
    ltf_maps = load_ltf_maps(env_yamls)

    rhcr_maps.mkdir(parents=True, exist_ok=True)
    created = []
    for map_name, home_count in sorted(targets.items()):
        if map_name not in ltf_maps:
            print(f"[skip] map not found in env yaml: {map_name}")
            continue
        grid = ltf_maps[map_name]
        rows = len(grid)
        cols = len(grid[0])
        out_rows, num_endpoints, num_homes = build_rhcr_map(grid, map_name, home_count)
        out_path = rhcr_maps / f"{args.prefix}{map_name}.map"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"{rows},{cols}\n")
            f.write(f"{num_endpoints}\n")
            f.write(f"{num_homes}\n")
            f.write(f"{int(args.max_time)}\n")
            for r in out_rows:
                f.write(r + "\n")
        created.append((map_name, out_path, rows, cols, num_endpoints, num_homes))
        print(
            f"[ok] {map_name} -> {out_path.name} "
            f"({rows}x{cols}, endpoints={num_endpoints}, homes={num_homes})"
        )

    if not created:
        print("No maps created.")
        return
    print(f"\nCreated {len(created)} RHCR map files under: {rhcr_maps}")
    print("Run RHCR once per map to auto-generate *_heuristics_table.txt.")


if __name__ == "__main__":
    main()
