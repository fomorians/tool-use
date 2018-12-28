#!/usr/bin/env bash
ENV_NAME=$1
SEED=$2
DESC=$3
PROJECT="tool-use"

JOB_NAME=$(echo "${ENV_NAME}_${SEED}_${DESC}" | tr "-" "_")
JOB_DIR="gs://$PROJECT-jobs/$ENV_NAME/$SEED/$DESC"

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