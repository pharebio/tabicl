#!/bin/bash
# Rung 4 of the column/row-attention ablation: train ProjectionEncoder against a FROZEN
# pretrained ICL stage. See docs/design/no_col_row_attention.md.
#
# What this asks: can the pretrained in-context-learning stage read a row representation
# that no attention produced? Only the ~0.6 M-parameter encoder is trainable; the 27.3 M
# ICL parameters are the published ones and are frozen, which is what makes the answer
# interpretable. Gradients still flow *through* the ICL stage to reach the encoder, so
# its activations are stored and memory is dominated by it, not by the encoder.
#
# Every ICL-shaping flag below is copied from train_v2_reg_stage3.sh, which produced the
# checkpoint being loaded. They must match or the pretrained weights will not load --
# the trainer raises rather than silently starting from noise.
#
# Differences from stage 3, all deliberate:
#   --row_encoder projection   neither attention stage is built
#   --min_features = --max_features = --proj_max_features
#                              The prior zero-pads to max_features behind a per-table d
#                              that the attention stages mask exactly. The projection
#                              encoder has no d and no mask, so a padding zero and a real
#                              zero are indistinguishable to it. Training at the single
#                              deployment width avoids spending capacity on widths this
#                              model can never serve anyway (its input layer is H-sized).
#   --freeze_icl True          the point of the experiment
#   --muon False, --lr 1e-3    the encoder is fresh, not being finetuned; stage 3's 2e-5
#                              is a finetuning rate for pretrained weights
#   --max_seq_len 10000        below stage 3's 60000. Length matters less here than it did
#                              for stage 3: the encoder is a per-row map, so it is
#                              structurally length-agnostic, and the only stage that sees
#                              length is frozen and was pretrained up to 60000.
#   --micro_batch_size 4       stage 3 used 1 across 4 GPUs. One card has room here because
#   --recompute False          the two attention stages are gone -- the ablation no longer
#                              builds a (T, H+4, 128) tensor to attend over, so gradient
#                              checkpointing buys little and costs ~30% throughput.
#
# PYTHONNOUSERSITE is mandatory: ~/.local/lib/python3.12/site-packages otherwise enters
# sys.path ahead of this environment's own packages.

set -euo pipefail

WIDTH=${WIDTH:-62}                  # deployment feature width (tabicl_monroe_pca62)
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
# Must be a multiple of MICRO_BATCH_SIZE: with --seq_len_per_gp True every group draws its
# own sequence length, and a micro batch must be of one length (_run.py:673). Stage 3 used
# 1 for both because it put one dataset on each of 4 GPUs.
BATCH_SIZE_PER_GP=${BATCH_SIZE_PER_GP:-$MICRO_BATCH_SIZE}
RECOMPUTE=${RECOMPUTE:-False}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-10000}
N_JOBS=${N_JOBS:-16}
SAVE_TEMP_EVERY=${SAVE_TEMP_EVERY:-50}
SAVE_PERM_EVERY=${SAVE_PERM_EVERY:-1000}
CKPT_DIR=${CKPT_DIR:-/data/andrew/admet/tabicl_rung4/pca${WIDTH}}
PYTHON=${PYTHON:-/data/andrew/envs/tabicl-train/bin/python}
STOCK_CKPT=${STOCK_CKPT:-$HOME/.cache/huggingface/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt}

export PYTHONNOUSERSITE=1

mkdir -p "$CKPT_DIR"

# Load the pretrained ICL weights only on the first launch. Once CKPT_DIR holds this
# run's own checkpoints the trainer resumes from the latest with optimizer and scheduler
# state, and --checkpoint_path would override that.
RESUME_ARGS="--checkpoint_path $STOCK_CKPT --only_load_model True"
if ls "$CKPT_DIR"/step-*.ckpt >/dev/null 2>&1; then
    RESUME_ARGS=""
fi

"$PYTHON" -m tabicl.train \
            --wandb_log False \
            --device cuda \
            --dtype float32 \
            --np_seed 43 \
            --torch_seed 43 \
            --max_steps "$MAX_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --micro_batch_size "$MICRO_BATCH_SIZE" \
            --lr 1e-3 \
            --muon False \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.02 \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 1.0 \
            --regression_method quantile \
            --num_quantiles 999 \
            --prior_type graph_scm \
            --prior_device cpu \
            --n_jobs "$N_JOBS" \
            --batch_size_per_gp "$BATCH_SIZE_PER_GP" \
            --min_features "$WIDTH" \
            --max_features "$WIDTH" \
            --min_seq_len 400 \
            --max_seq_len "$MAX_SEQ_LEN" \
            --log_seq_len True \
            --min_train_size 0.79 \
            --max_train_size 0.81 \
            --seq_len_per_gp True \
            --graph_noise False \
            --filter_unpredictable_graphs True \
            --filter_unpredictable_datasets True \
            --allow_act_warping False \
            --min_n_nodes 2 \
            --max_n_nodes 32 \
            --cauchy_dag_offset 0.0 \
            --row_encoder projection \
            --proj_max_features "$WIDTH" \
            --freeze_icl True \
            --embed_dim 128 \
            --row_num_cls 4 \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            --icl_ssmax True \
            --ssmax_type qassmax-mlp-elementwise \
            --ff_factor 2 \
            --norm_first True \
            --zero_init False \
            --use_flash_attn3 True \
            --norm_type layernorm_nobias \
            --recompute "$RECOMPUTE" \
            --checkpoint_dir "$CKPT_DIR" \
            $RESUME_ARGS \
            --save_temp_every "$SAVE_TEMP_EVERY" \
            --save_perm_every "$SAVE_PERM_EVERY"
