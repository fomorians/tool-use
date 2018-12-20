#!/usr/bin/env bash
ENVS='MountainCarContinuous-v0 Pendulum-v0 LunarLanderContinuous-v2 BipedalWalker-v2'
for ENV in $ENVS
do
    for SEED in {1..5}
    do
        python -m tool_use.main --job-dir "$HOME/jobs/tool_use/$ENV/$SEED/discount" --env-name $ENV --seed $SEED --use-discount
        python -m tool_use.main --job-dir "$HOME/jobs/tool_use/$ENV/$SEED/custom" --env-name $ENV --seed $SEED
    done
done
