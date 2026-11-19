#!/bin/bash
# Parallel version of `sea_pilot.sh train`: one worker per listed GPU slot, each claiming
# (program, seed) runs through a lock directory. Safe to restart; finished runs are skipped.
# Usage: SEEDS="0" BUF=cpu runs/sea_pilot_workers.sh <task> "<gpu> <gpu> ..."   (repeat a GPU for 2 runs on it)
set -uo pipefail
TASK=$1
GPUS=$2
NAME=${TASK#*_}
ROOT=${SEA_ROOT:-logdir/pilot}/$NAME
STEPS=${STEPS:-5.1e5}
SEEDS=${SEEDS:-"0 1"}
REP=${REP:-dreamer}
BUF=${BUF:-cuda:0}
EXTRA=${EXTRA:-}
case $TASK in
    metaworld_*) ENV=metaworld ;;
    dmc_*) ENV=dmc_vision ;;
    *) echo "unknown suite in $TASK"; exit 1 ;;
esac
mkdir -p $ROOT/runs $ROOT/locks $ROOT/worker_logs

worker() {
    local gpu=$1 slot=$2
    for seed in $SEEDS; do
        for prog in $ROOT/programs/*.py; do
            local id=$(basename $prog .py)_s$seed
            local out=$ROOT/runs/$id
            [ -f $out/latest.pt ] && continue
            mkdir $ROOT/locks/$id 2>/dev/null || continue
            echo "$(date +%F_%T) slot $slot gpu $gpu start $id"
            rm -rf $out
            CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu python train.py \
                env=$ENV env.task=$TASK env.steps=$STEPS model.rep_loss=$REP device=cuda:0 \
                buffer.storage_device=$BUF $EXTRA seed=$seed logdir=$out \
                sea.enabled=True sea.mode=fixed sea.programs=$prog sea.max_dim=96 \
                > $ROOT/worker_logs/$id.log 2>&1
            echo "$(date +%F_%T) slot $slot gpu $gpu done $id exit=$?"
            [ -f $out/latest.pt ] || rmdir $ROOT/locks/$id   # failed: release so it can be retried
        done
    done
}

slot=0
for gpu in $GPUS; do
    worker $gpu $slot &
    slot=$((slot + 1))
    sleep 20
done
wait
echo "$(date +%F_%T) all workers finished"
