#!/usr/bin/env python
"""Time TabICL's three stages separately at a library-inference shape.

Calls col_embedder, row_interactor and icl_predictor one after another on the same
tensors, synchronizing between them, so each stage's wall clock and the pipeline's peak
GPU memory are attributed per stage rather than to the fused forward pass. Written for
the no-col-row-attention branch: it is the before/after instrument for replacing the
first two stages, and the source of the tables in
docs/design/no_col_row_attention.md.

One row per feature width. Column `icl_s` should be flat across widths -- the ICL stage
does not see feature count -- and `col_s` + `row_s` is what grows.

A fixed InferenceConfig is NOT used by default: the point is to measure what the
shipped default does. Pass --safety_factor to pin batch sizing when comparing runs.

    python benchmarks/stage_timing/measure_stage_timing.py
    python benchmarks/stage_timing/measure_stage_timing.py --widths 62 --repeats 5
"""

import argparse
import pathlib
import statistics
import time

import pandas as pd
import torch

from tabicl._model.inference_config import InferenceConfig
from tabicl._model.tabicl import TabICL

HERE = pathlib.Path(__file__).parent
DEFAULT_WIDTHS = [23, 62, 100, 200, 720]
DEFAULT_CKPT = "tabicl-regressor-v2-20260212.ckpt"
HF_REPO = "jingang/TabICL"


def load_model(checkpoint: str, device: str) -> TabICL:
    """Build TabICL from a checkpoint path, or download `checkpoint` from the Hub by name."""
    path = pathlib.Path(checkpoint)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = pathlib.Path(hf_hub_download(repo_id=HF_REPO, filename=checkpoint))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])
    config.pop("recompute", None)
    model = TabICL(**config)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device)


def time_stages(model, n_context, n_test, width, config, device):
    """One forward pass, timed per stage. Returns (col_s, row_s, icl_s, peak_gb)."""
    total = n_context + n_test
    X = torch.randn(1, total, width, device=device)
    y = torch.randn(1, n_context, device=device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        embeddings = model.col_embedder(X, y_train=y, embed_with_test=False, mgr_config=config.COL_CONFIG)
        sync()
        t1 = time.perf_counter()
        representations = model.row_interactor(embeddings, mgr_config=config.ROW_CONFIG)
        del embeddings
        sync()
        t2 = time.perf_counter()
        model.icl_predictor(
            representations,
            y_train=y,
            return_logits=True,
            softmax_temperature=0.9,
            mgr_config=config.ICL_CONFIG,
        )
        sync()
        t3 = time.perf_counter()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else float("nan")
    del X, y, representations
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    model.clear_cache()
    return t1 - t0, t2 - t1, t3 - t2, peak_gb


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--n_context", type=int, default=5626, help="context rows, one 5x5 PPB fold")
    parser.add_argument("--n_test", type=int, default=50000, help="test rows per call, the library chunk")
    parser.add_argument("--repeats", type=int, default=3, help="timed passes per width; the median is kept")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--safety_factor", type=float, default=None,
                        help="pin batch sizing on every stage; omit to use the shipped default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=pathlib.Path, default=HERE / "stage_timing.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"[1] loading {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, args.device)
    per_stage = {
        name: sum(p.numel() for p in getattr(model, name).parameters())
        for name in ["col_embedder", "row_interactor", "icl_predictor"]
    }
    total_params = sum(per_stage.values())
    for name, count in per_stage.items():
        print(f"    {name:16s} {count / 1e6:7.3f} M  {100 * count / total_params:5.1f}%", flush=True)

    config = InferenceConfig()
    if args.safety_factor is not None:
        for stage in [config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG]:
            stage.safety_factor = args.safety_factor
        print(f"    batch sizing pinned at safety_factor={args.safety_factor}", flush=True)

    print(f"[2] warming up ({args.warmup} passes at width {args.widths[0]})", flush=True)
    for _ in range(args.warmup):
        time_stages(model, args.n_context, args.n_test, args.widths[0], config, args.device)

    print(f"[3] timing {len(args.widths)} widths, {args.repeats} passes each", flush=True)
    rows = []
    for width in args.widths:
        runs = [time_stages(model, args.n_context, args.n_test, width, config, args.device)
                for _ in range(args.repeats)]
        col = statistics.median(r[0] for r in runs)
        row = statistics.median(r[1] for r in runs)
        icl = statistics.median(r[2] for r in runs)
        peak = max(r[3] for r in runs)
        total = col + row + icl
        rows.append({
            "n_features": width,
            "n_context": args.n_context,
            "n_test": args.n_test,
            "col_s": round(col, 4),
            "row_s": round(row, 4),
            "icl_s": round(icl, 4),
            "total_s": round(total, 4),
            "col_row_share": round((col + row) / total, 4),
            "peak_gpu_gb": round(peak, 2),
        })
        print(f"    H={width:4d}  col={col:7.3f}s  row={row:7.3f}s  icl={icl:7.3f}s  "
              f"total={total:7.3f}s  col+row={100 * (col + row) / total:5.1f}%  peak={peak:6.2f} GB",
              flush=True)

    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    print(f"[4] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
