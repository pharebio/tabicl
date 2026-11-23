#!/usr/bin/env python
"""Draw inference_speed_per_1M.png from inference_speed_per_1M.csv.

One stacked horizontal bar per model: featurization, then model (context load +
predict), in hours to score 1M compounds with the 25-model ensemble. Measured on
yu14M library shards; the rows_measured column says how many rows each number comes
from, and the hours are per 1M regardless.

Two panels. The top one holds every model on one linear axis, which is what shows
that the 2,048-column feature sets are off the table at 40-50 h. On that axis the
models anyone would actually deploy are slivers 1-7% of the width, so the bottom
panel repeats the ones under the DETAIL_MAX cut on an expanded axis.

    python benchmarks/ppb_inference/plot_inference_speed.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = pathlib.Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C_FEAT, C_MODEL = "#eb6834", "#2a78d6"
DETAIL_MAX = 4.0  # hours; models at or under this are repeated in the lower panel

# Second label line for the models whose featurization/model split needs a caveat.
NOTES = {
    "chemprop": "25 CheMeleon encoder passes, inside model time",
    "tabicl_chemeleon": "frozen CheMeleon pass",
    "tabicl_chemeleon_pca720": "CheMeleon pass + PCA projection",
    "tabicl_chemeleon_pca200": "CheMeleon pass + PCA projection",
    "tabicl_chemeleon_pca100": "CheMeleon pass + PCA projection",
    "tabicl_monroe": "conformers + GRIT encoder",
    "chemprop_monroe": "Monroe pass, then chemprop with 720-d descriptors",
    "chemprop_monroe_pca200": "Monroe pass + PCA projection, then chemprop",
    "tabicl_monroe_pca200": "Monroe pass + PCA projection",
    "tabicl_monroe_pca62": "Monroe pass + PCA projection",
    "tabicl_monroe_pca62_noattn": "Monroe pass + PCA; column/row attention removed, ICL kept",
    "tabicl_monroe_pca23": "Monroe pass + PCA projection",
    "tabicl_monroe_pca100": "Monroe pass + PCA projection",
    "rf_monroe": "Monroe pass, then a 100-tree forest",
    "rf_monroe_pca200": "Monroe pass + PCA projection, then a forest",
    "rf_monroe_pca100": "Monroe pass + PCA projection, then a forest",
}


def draw(ax, df, label_room, notes=True):
    """Stacked bars + right-hand labels for `df`, in the order already applied.

    The bars are the COMMON chunk (50,000) so every model is drawn under one setting.
    Per-model best chunks are reported in the label, not in the bar: they are not
    comparable with each other, since for the narrow feature sets the sweep's best was
    50,000 itself -- the edge of the range, with the curve still falling -- while for
    chemeleon the best was a true interior minimum at 10,000.
    """
    y = range(len(df))
    ax.barh(y, df.featurize_hours_per_1M, color=C_FEAT, height=0.6, zorder=3)
    ax.barh(y, df.predict_hours_per_1M_at_chunk50k, left=df.featurize_hours_per_1M,
            color=C_MODEL, height=0.6, zorder=3)

    xmax = df.total_hours_per_1M_at_chunk50k.max()
    for i, r in df.reset_index(drop=True).iterrows():
        label = (f"{r.total_hours_per_1M_at_chunk50k:.2f} h   "
                 f"(featurization {r.featurize_hours_per_1M:.2f} + "
                 f"model {r.predict_hours_per_1M_at_chunk50k:.2f})")
        bits = []
        if pd.notna(r.predict_chunk) and r.total_hours_per_1M_at_chunk50k > r.total_hours_per_1M * 1.01:
            bits.append(f"{r.total_hours_per_1M:.1f} h at its best measured chunk "
                        f"({int(r.predict_chunk):,})")
        note = NOTES.get(r.model) if notes else None
        if note:
            bits.append(note)
        second = "   ·   ".join(bits)
        ax.text(r.total_hours_per_1M_at_chunk50k + xmax * 0.012, i,
                label if not second else label + "\n" + second,
                va="center", ha="left", fontsize=7.5, color=INK2, zorder=4)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df.model, fontsize=9, color=INK)
    ax.set_xlim(0, xmax * label_room)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def main():
    df = pd.read_csv(HERE / "inference_speed_per_1M.csv")
    df = df.sort_values("total_hours_per_1M_at_chunk50k", ascending=False).reset_index(drop=True)
    detail = df[df.total_hours_per_1M_at_chunk50k <= DETAIL_MAX].reset_index(drop=True)

    rows = len(df) + len(detail)
    fig, (ax_all, ax_det) = plt.subplots(
        2, 1, figsize=(10.5, 0.52 * rows + 2.6), dpi=200,
        gridspec_kw={"height_ratios": [len(df), len(detail)], "hspace": 0.26})
    fig.patch.set_facecolor("white")

    draw(ax_all, df, label_room=1.62)
    ax_all.set_title("PPB inference cost per 1M compounds, every model at predict chunk 50,000",
                     fontsize=11.5, color=INK, loc="left", pad=12)
    ax_all.set_xlabel("Hours to score 1M compounds (25-model ensemble)", fontsize=9, color=INK)

    draw(ax_det, detail, label_room=1.78, notes=False)
    ax_det.set_title(f"Detail: the {len(detail)} models under {DETAIL_MAX:.0f} h, same bars on an "
                     "expanded axis", fontsize=9.5, color=INK, loc="left", pad=8)
    ax_det.set_xlabel("Hours to score 1M compounds (25-model ensemble)", fontsize=9, color=INK)

    # above the axes: the long bars reach the bottom-right corner, so an in-axes legend
    # collides with their value labels
    fig.legend(handles=[Patch(color=C_FEAT, label="Featurization (graph building, plus any feature pass the model needs)"),
                        Patch(color=C_MODEL, label="Model (load/context + predict)")],
               loc="upper right", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.995, 0.995))

    fig.text(0.01, 0.012,
             "Measured on yu14M library shards, one RTX 6000 Ada (48 GB), TabICL without the KV cache. "
             "RF runs on 32 CPU cores.\n"
             "chemprop is fine-tuned from the CheMeleon foundation model, so its encoder runs once per "
             "fold inside model time; the TabICL CheMeleon\nfeature sets instead use a single frozen "
             "CheMeleon pass, which is counted as featurization. Plain chemprop's featurization is graph "
             "building alone; the two chemprop\nMonroe rows take the embedding as molecule-level "
             "descriptors, so theirs is graph building plus the same Monroe pass the rf_monroe and\n"
             "tabicl_monroe rows are charged for.\n"
             "TabICL re-encodes its context on every predict call, so cost depends strongly on chunk "
             "size. The bars hold that setting fixed at 50,000 for\nevery model, which is the controlled "
             "comparison; a model's own best chunk over the 2,500-50,000 sweep is given in its label where "
             "it is\nfaster. Those per-model bests are not comparable with each other: for the narrow "
             "feature sets the best was 50,000 itself, the edge of\nthe sweep with the curve still falling, "
             "while for chemeleon it was a true interior minimum at 10,000. Chunk size shifts library\n"
             "predictions slightly (mean 0.0017 on a target of sd 0.53) but not the held-out metrics, "
             "which are predicted unchunked.\n"
             "Featurization is charged over both measured shards for every model. Monroe's is conformer "
             "generation plus the GRIT encoder, and the two\nshards differ (274 vs 185 molecules/s), so a "
             "shard-0-only figure understates it by 24%.",
             fontsize=7, color=INK2)
    # left=0.205 fits the longest model name (tabicl_monroe_pca62_noattn) at fontsize 9;
    # 0.175 clipped it.
    # Fixed margins in inches rather than tight_layout, which cannot place two panels with
    # out-of-axes labels and silently clipped the model names off the left edge.
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.205, right=0.995, top=1 - 0.92 / h, bottom=1.95 / h)
    out = HERE / "inference_speed_per_1M.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
