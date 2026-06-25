#!/usr/bin/env python3

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


MAP_FILE = "maps/wfi_warehouse.map"
PATHS_FILE = "/home/shiqi/masterarbeit/exp/wfi_learned_seed0/paths.txt"
MAX_STEPS = None

FRAME_STRIDE = 1
INTERVAL_MS = 100
AGENT_SIZE = 30
SHOW_AGENT_IDS = False

def detect_map_type(path):
    """Return 'rhcr_graph_grid' or 'ascii_map'."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = [f.readline().strip().replace("\r", "") for _ in range(10)]
    head = [ln for ln in head if ln != ""]

    if any(ln.lower().startswith("grid size") for ln in head) or any("id,type,station" in ln.lower() for ln in head):
        return "rhcr_graph_grid"

    if len(head) >= 1 and "," in head[0]:
        parts = head[0].split(",")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return "ascii_map"

    raise ValueError(f"Unknown map format: {path}. First lines: {head[:5]}")


def load_rhcr_graph_grid(path):
    """
    RHCR .grid (CSV graph):
    Grid size (x, y)
    W,H
    id,type,station,x,y, ...
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip().replace("\r", "") for ln in f if ln.strip() != ""]

    width = height = None
    for i, ln in enumerate(lines[:10]):
        if ln.lower().startswith("grid size"):
            parts = [p.strip() for p in lines[i + 1].split(",")]
            width = int(parts[0])
            height = int(parts[1])
            break
    if width is None or height is None:
        raise ValueError(f"Cannot find grid size in {path}")

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("id,type,station"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot find CSV header in {path}")

    id2xy = {}
    id2type = {}

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(csv_text.splitlines())

    for row in reader:
        vid = int(row["id"])
        ntype = row["type"].strip()
        x = int(row["x"])
        y = int(row["y"])
        id2xy[vid] = (x, y)
        id2type[vid] = ntype

    occ = np.zeros((height, width), dtype=np.uint8)
    for vid, (x, y) in id2xy.items():
        if id2type.get(vid, "").lower() == "obstacle":
            occ[y, x] = 1

    return width, height, occ, id2xy


def load_ascii_map(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_all = [ln.rstrip("\n").replace("\r", "") for ln in f]

    raw_all = [ln for ln in raw_all if ln.strip() != ""]

    rows_str, cols_str = [p.strip() for p in raw_all[0].split(",")]
    rows, cols = int(rows_str), int(cols_str)
    H, W = rows, cols

    allowed = set(".@reREsgSG0123456789-_*# ")
    start = None
    for i in range(1, len(raw_all)):
        ln = raw_all[i]
        if len(ln) < max(5, W - 2):
            continue
        bad = sum(ch not in allowed for ch in ln)
        if bad <= max(2, len(ln) // 20):
            if ('.' in ln) or ('@' in ln) or ('r' in ln) or ('e' in ln) or ('R' in ln) or ('E' in ln):
                start = i
                break

    if start is None:
        raise ValueError(f"ASCII map {path}: cannot locate grid start automatically.")

    grid_lines = raw_all[start:start + H]
    if len(grid_lines) < H:
        print(f"[WARN] Map header says H={H}, but only found {len(grid_lines)} grid lines. "
            f"Using H={len(grid_lines)} for visualization.")
        H = len(grid_lines)

    grid_lines = [ln.ljust(W, ".")[:W] for ln in grid_lines]

    occ = np.zeros((H, W), dtype=np.uint8)
    id2xy = {}
    for y in range(H):
        for x in range(W):
            ch = grid_lines[y][x]
            vid = y * W + x
            id2xy[vid] = (x, y)
            if ch == "@":
                occ[y, x] = 1

    return W, H, occ, id2xy



def load_paths(paths_file):
    with open(paths_file, "r", encoding="utf-8", errors="ignore") as f:
        n = int(f.readline().strip())
        lines = [f.readline().strip() for _ in range(n)]

    max_t = 0
    tv_lists = []
    for ln in lines:
        segs = [s for s in ln.split(";") if s]
        tv = []
        for s in segs:
            parts = s.split(",")
            if len(parts) < 3:
                continue
            v = int(parts[0])
            t = int(parts[2])
            tv.append((t, v))
            max_t = max(max_t, t)
        tv.sort(key=lambda x: x[0])
        tv_lists.append(tv)

    T = max_t + 1
    paths = np.full((n, T), -1, dtype=int)

    for i, tv in enumerate(tv_lists):
        if not tv:
            continue
        for (t, v) in tv:
            if 0 <= t < T:
                paths[i, t] = v

        known = np.where(paths[i] != -1)[0]
        if len(known) == 0:
            continue
        fk = known[0]
        paths[i, :fk] = paths[i, fk]
        for t in range(fk + 1, T):
            if paths[i, t] == -1:
                paths[i, t] = paths[i, t - 1]

    return paths


def main():
    mtype = detect_map_type(MAP_FILE)
    if mtype == "rhcr_graph_grid":
        W, H, occ, id2xy = load_rhcr_graph_grid(MAP_FILE)
    elif mtype == "ascii_map":
        W, H, occ, id2xy = load_ascii_map(MAP_FILE)
    else:
        raise RuntimeError("Unexpected map type")

    paths = load_paths(PATHS_FILE)
    N, T = paths.shape
    if MAX_STEPS is not None:
        T = min(T, MAX_STEPS)
        paths = paths[:, :T]

    print("map type:", mtype)
    print("map (W,H):", (W, H))
    print("paths (N,T):", (N, T))
    print("known nodes:", len(id2xy))
    print("paths v_max:", int(paths.max()))

    sample = paths.flatten()[:: max(1, (paths.size // 2000))]
    missing = sum(int(v) not in id2xy for v in sample)
    print("sample missing vertex IDs:", missing, "/", len(sample))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    ax.imshow(occ, interpolation="nearest")

    scat = ax.scatter([], [], s=AGENT_SIZE, c="red", edgecolors="white", linewidths=0.3)

    texts = []
    if SHOW_AGENT_IDS:
        for i in range(N):
            texts.append(ax.text(0, 0, str(i), fontsize=8))

    frames = list(range(0, T, FRAME_STRIDE))

    def update(t):
        xs, ys = [], []
        for i in range(N):
            vid = int(paths[i, t])
            if vid in id2xy:
                x, y = id2xy[vid]
                xs.append(x); ys.append(y)
            else:
                xs.append(np.nan); ys.append(np.nan)

        scat.set_offsets(np.c_[xs, ys])
        if SHOW_AGENT_IDS:
            for i, txt in enumerate(texts):
                txt.set_position((xs[i], ys[i]))

        ax.set_title(f"t={t}  stride={FRAME_STRIDE}")
        return (scat, *texts) if SHOW_AGENT_IDS else (scat,)

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
        repeat=True,
    )
    fig._ani = ani
    plt.show()


if __name__ == "__main__":
    main()
