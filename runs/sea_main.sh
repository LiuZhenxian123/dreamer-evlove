#!/bin/bash
# Main comparison on one task. Usage: runs/sea_main.sh <method> <task> <seed> [gpu]
#   base      no privileged supervision
#   raw       predict the raw privileged state            (needs runs/sea_pilot.sh candidates first)
#   random    random linear mixtures of the raw state     (same)
#   oneshot   an LLM program used as generated, with no scoring or feedback (oneshot_00..02 by seed)
#   evolve    LLM evolution during training, selected by the training-free fitness
#   evolve_lib  evolve + cross-task library at $SEA_LIBRARY
set -euo pipefail

METHOD=$1
TASK=$2
SEED=$3
GPU=${4:-0}
NAME=${TASK#*_}
OUT=${SEA_OUT:-logdir/main}/${NAME}/${METHOD}_s$SEED
PROGS=${SEA_ROOT:-logdir/pilot}/$NAME/programs
STEPS=${STEPS:-1.01e6}
REP=${REP:-dreamer}
MODEL=${SEA_LLM_MODEL:-gpt-4o}
case $TASK in
    metaworld_*) ENV=metaworld; EXTRA="env.configs=train env.eval_configs=test" ;;
    dmc_*) ENV=dmc_vision; EXTRA="" ;;
    *) echo "unknown suite in $TASK"; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
COMMON="env=$ENV env.task=$TASK env.steps=$STEPS model.rep_loss=$REP device=cuda:0 buffer.storage_device=cuda:0 seed=$SEED logdir=$OUT trainer.save_every=2.5e5 $EXTRA"

case $METHOD in
base)    python train.py $COMMON ;;
raw)     python train.py $COMMON sea.enabled=True sea.mode=fixed sea.programs=$PROGS/raw.py sea.max_dim=96 ;;
random)  python train.py $COMMON sea.enabled=True sea.mode=fixed sea.programs=$PROGS/random_0$((SEED % 3)).py ;;
oneshot) python train.py $COMMON sea.enabled=True sea.mode=fixed sea.programs=$PROGS/oneshot_0$((SEED % 3)).py ;;
evolve)  python train.py $COMMON sea.enabled=True sea.mode=evolve sea.llm.model=$MODEL ;;
evolve_lib)
    python train.py $COMMON sea.enabled=True sea.mode=evolve sea.llm.model=$MODEL sea.library=${SEA_LIBRARY:?set SEA_LIBRARY} ;;
*) echo "unknown method $METHOD"; exit 1 ;;
esac

# Zero-shot evaluation under shift (the periodic eval during training already uses the
# held-out layouts for Meta-World; this adds the DMC dynamics grid).
if [ $ENV = dmc_vision ]; then
    python eval.py --logdir $OUT --episodes 30 --device cuda:0 --shifts \
        motor=0.5 motor=0.75 motor=1.25 motor=1.5 friction=0.5 friction=0.75 friction=1.25 friction=1.5 \
        motor=0.5,friction=0.5 motor=0.75,friction=0.75 motor=1.25,friction=1.25 motor=1.5,friction=1.5
else
    python eval.py --logdir $OUT --episodes 50 --device cuda:0 --configs test
fi
