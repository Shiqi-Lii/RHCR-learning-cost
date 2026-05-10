#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _read_input(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows, cols, history_len = map(int, f.readline().strip().split())
        size = rows * cols
        obstacles = np.fromstring(f.readline().strip(), sep=" ", dtype=np.float32)
        if obstacles.size != size:
            raise ValueError("invalid obstacle size in input")
        obstacles = obstacles.reshape(rows, cols)
        occ = []
        v = []
        h = []
        for _ in range(history_len):
            occ_i = np.fromstring(f.readline().strip(), sep=" ", dtype=np.float32)
            v_i = np.fromstring(f.readline().strip(), sep=" ", dtype=np.float32)
            h_i = np.fromstring(f.readline().strip(), sep=" ", dtype=np.float32)
            if occ_i.size != size or v_i.size != size or h_i.size != size:
                raise ValueError("invalid frame size in input")
            occ.append(occ_i.reshape(rows, cols))
            v.append(v_i.reshape(rows, cols))
            h.append(h_i.reshape(rows, cols))
    return rows, cols, obstacles, np.stack(occ, axis=0), np.stack(v, axis=0), np.stack(h, axis=0)


def _infer_model(state_dict):
    use_model_v2 = any(".block." in key for key in state_dict.keys())
    conv0 = state_dict["encoder.0.weight"]
    in_channels = int(conv0.shape[1])
    hidden_channels = int(conv0.shape[0])
    layer_ids = set()
    for k in state_dict.keys():
        if k.startswith("convlstm.cell_list.") and ".conv.weight" in k:
            layer_ids.add(int(k.split(".")[2]))
    num_layers = max(layer_ids) + 1 if layer_ids else 1
    kernel_sizes = []
    for i in range(num_layers):
        w = state_dict[f"convlstm.cell_list.{i}.conv.weight"]
        kernel_sizes.append(int(w.shape[-1]))
    return use_model_v2, in_channels, hidden_channels, num_layers, kernel_sizes


def _gaussian_kernel(ksize: int, sigma: float, device, dtype):
    r = ksize // 2
    coords = torch.arange(-r, r + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k / (k.sum() + 1e-12)
    return k.view(1, 1, ksize, ksize)


@torch.no_grad()
def _masked_gaussian_smooth(cost: torch.Tensor, mask: torch.Tensor, ksize: int, sigma: float):
    pad = ksize // 2
    kernel = _gaussian_kernel(ksize, sigma, cost.device, cost.dtype)
    num = F.conv2d(cost * mask, kernel, padding=pad)
    den = F.conv2d(mask, kernel, padding=pad)
    return (num / (den + 1e-6)) * mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--gaussian_sigma", type=float, required=True)
    parser.add_argument("--pred_bias", type=float, required=True)
    parser.add_argument("--gaussian_ksize", type=int, default=9)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ltf_root = repo_root / "learn-to-follow"
    if str(ltf_root) not in sys.path:
        sys.path.insert(0, str(ltf_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(state_dict, dict):
        raise ValueError("invalid checkpoint format")

    use_model_v2, in_channels, hidden_channels, num_layers, kernel_sizes = _infer_model(state_dict)
    if use_model_v2:
        from costmap.model_v2 import ConvLSTMModel
    else:
        from costmap.model import ConvLSTMModel
    model = ConvLSTMModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_sizes=kernel_sizes,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def infer_once(rows, cols, obstacles, occ, v, h):
        mask = (obstacles == 0).astype(np.float32)
        channels = []
        if in_channels == 4:
            channels = [obstacles, occ, v, h]
        elif in_channels == 3:
            channels = [occ, v, h]
        elif in_channels == 2:
            channels = [obstacles, occ]
        elif in_channels == 1:
            channels = [occ]
        else:
            raise ValueError(f"unsupported inferred in_channels={in_channels}")

        frames = []
        for t in range(occ.shape[0]):
            c = []
            for item in channels:
                if item.ndim == 2:
                    c.append(item)
                else:
                    c.append(item[t])
            frames.append(np.stack(c, axis=0))
        x = np.stack(frames, axis=0)[None]
        x_t = torch.from_numpy(x).float().to(device)

        pred = model(x_t)[0, 0]
        pred = torch.clamp(pred, min=0.0)
        mask_t = torch.from_numpy(mask).to(device=device, dtype=pred.dtype).unsqueeze(0).unsqueeze(0)
        if args.gaussian_ksize > 1 and args.gaussian_ksize % 2 == 1 and args.gaussian_sigma > 0:
            orig_max = pred.max()
            pred = _masked_gaussian_smooth(pred.unsqueeze(0).unsqueeze(0), mask_t, args.gaussian_ksize, args.gaussian_sigma)[0, 0]
            pred = pred * (orig_max / (pred.max() + 1e-6))
        if args.normalize:
            pred = pred / (pred.max() + 1e-6)
        pred = (pred + args.pred_bias) * mask_t[0, 0]
        pred = pred * args.weight
        return pred.detach().cpu().numpy().reshape(rows * cols)

    if args.server:
        while True:
            header = sys.stdin.readline()
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            if header == "QUIT":
                break
            parts = header.split()
            if len(parts) != 4 or parts[0] != "INFER":
                print("ERR invalid header", flush=True)
                continue
            rows = int(parts[1]); cols = int(parts[2]); history_len = int(parts[3])
            size = rows * cols
            obstacles = np.fromstring(sys.stdin.readline().strip(), sep=" ", dtype=np.float32).reshape(rows, cols)
            occ = np.zeros((history_len, rows, cols), dtype=np.float32)
            v = np.zeros((history_len, rows, cols), dtype=np.float32)
            h = np.zeros((history_len, rows, cols), dtype=np.float32)
            for t in range(history_len):
                occ[t] = np.fromstring(sys.stdin.readline().strip(), sep=" ", dtype=np.float32).reshape(rows, cols)
            for t in range(history_len):
                v[t] = np.fromstring(sys.stdin.readline().strip(), sep=" ", dtype=np.float32).reshape(rows, cols)
            for t in range(history_len):
                h[t] = np.fromstring(sys.stdin.readline().strip(), sep=" ", dtype=np.float32).reshape(rows, cols)
            out = infer_once(rows, cols, obstacles, occ, v, h)
            print(f"OK {rows} {cols}", flush=True)
            print(" ".join(str(float(x)) for x in out), flush=True)
        return

    if not args.input or not args.output:
        raise ValueError("--input and --output are required when not running with --server")
    rows, cols, obstacles, occ, v, h = _read_input(Path(args.input))
    out = infer_once(rows, cols, obstacles, occ, v, h)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"{rows} {cols}\n")
        f.write(" ".join(str(float(x)) for x in out))
        f.write("\n")


if __name__ == "__main__":
    main()
