#!/usr/bin/env bash
ENV_NAME=$1
SEED=$2
DESC=$3

JOB_NAME=$(echo "${ENV_NAME}_${SEED}_${DESC}" | tr "-" "_")
JOB_DIR="gs://tool-use-jobs/$ENV_NAME/$SEED/$DESC"

PACKAGES="$HOME/Documents/pyoneer/dist/pyoneer-0.0.0.tar.gz,$HOME/Documents/gym_pycolab/dist/gym_pycolab-0.0.0.tar.gz,$HOME/Documents/gym_tool_use/dist/gym_tool_use-1.0.0.tar.gz"

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket "gs://tool-use-staging" \
    --package-path "$(pwd)/tool_use" \
    --job-dir $JOB_DIR \
    --packages $PACKAGES \
    --module-name "tool_use.main" \
    --config "$(pwd)/config.yaml" \
    --stream-logs \
    -- \
    --env-name $ENV_NAME \
    --seed $SEED
