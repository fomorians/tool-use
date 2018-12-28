#!/usr/bin/env bash
ENV_NAME='Pendulum-v0'
SEED=$(date +"%s")
PROJECT="tool-use"

JOB_NAME=$(echo "${ENV_NAME}_$SEED" | tr "-" "_")_$1
JOB_DIR="gs://$PROJECT-jobs/$ENV_NAME/$SEED/$1"

PACKAGES="$HOME/Documents/trfl/dist/trfl-1.0.tar.gz,$HOME/Documents/pyoneer/dist/pyoneer-0.0.0.tar.gz,$HOME/Documents/box2d-py/dist/box2d_py-2.3.8-cp35-cp35m-linux_x86_64.whl"

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket "gs://tool-use-staging" \
    --package-path "$(pwd)/tool_use" \
    --job-dir $JOB_DIR \
    --packages $PACKAGES \
    --module-name "tool_use.main" \
    --config "$(pwd)/config.yaml" \
    -- \
    --env-name $ENV_NAME \
    --seed $SEED