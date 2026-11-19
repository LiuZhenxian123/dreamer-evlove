#!/bin/bash
# Pilot: does the training-free fitness predict downstream return?
# Usage: runs/sea_pilot.sh <stage> <task> [gpu]      stage = collect | candidates | train | analyze
# Example: runs/sea_pilot.sh collect metaworld_pick-place 0
# See EXPERIMENTS.md for the protocol and the go/no-go rule.
set -euo pipefail

STAGE=$1
TASK=$2
GPU=${3:-0}
NAME=${TASK#*_}
ROOT=${SEA_ROOT:-logdir/pilot}/$NAME
STEPS=${STEPS:-5.1e5}          # per training run
N_ONESHOT=${N_ONESHOT:-30}     # LLM candidates
SEEDS=${SEEDS:-"0 1"}
REP=${REP:-dreamer}            # dreamer (DreamerV3, with decoder) | r2dreamer (decoder-free, faster)
BUF=${BUF:-cuda:0}             # replay storage; use cpu on shared GPUs (saves about 6 GB of GPU memory)
EXTRA=${EXTRA:-}               # extra train.py overrides
case $TASK in
    metaworld_*) ENV=metaworld ;;
    dmc_*) ENV=dmc_vision ;;
    *) echo "unknown suite in $TASK"; exit 1 ;;
esac
COMMON="env=$ENV env.task=$TASK env.steps=$STEPS model.rep_loss=$REP device=cuda:0 buffer.storage_device=$BUF $EXTRA"
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU

case $STAGE in
collect)
    # Plain agent, no semantic head. Saves scoring datasets at 20% and 50% of training.
    python train.py $COMMON seed=0 logdir=$ROOT/collect sea.dump_data_at=[0.2,0.5] sea.data_batches=48
    ;;
candidates)
    DATA=$(ls $ROOT/collect/datasets/step_*.pt | sort -V | head -1)
    python -m sea.cli baselines --data $DATA --out $ROOT/programs --random 3 --dim 8
    python -m sea.cli sample --data $DATA --task $TASK --n $N_ONESHOT --out $ROOT/programs
    python -m sea.cli score --data $DATA --task $TASK --programs $ROOT/programs --out $ROOT/scores.json
    ;;
train)
    # One run per (program, seed). raw.py needs more target slots than the default.
    for prog in $ROOT/programs/*.py; do
        for seed in $SEEDS; do
            out=$ROOT/runs/$(basename $prog .py)_s$seed
            [ -f $out/latest.pt ] && continue
            python train.py $COMMON seed=$seed logdir=$out \
                sea.enabled=True sea.mode=fixed sea.programs=$prog sea.max_dim=96
        done
    done
    ;;
analyze)
    METRIC=episode/eval_score
    [ $ENV = metaworld ] && METRIC=episode/eval_success
    python -m sea.cli rankcorr --scores $ROOT/scores.json --runs "$ROOT/runs/*" --metric $METRIC --out $ROOT/rankcorr.json
    ;;
*)
    echo "unknown stage $STAGE"; exit 1 ;;
esac
