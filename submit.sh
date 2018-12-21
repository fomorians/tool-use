#!/usr/bin/env bash
ENV_NAME=$1
SEED=$(date +"%s")
PROJECT="tool-use"

JOB_NAME=$(echo "${ENV_NAME}_$SEED" | tr "-" "_")
JOB_PATH=$(echo "${JOB_NAME}" | tr "[:upper:]" "[:lower:]" | tr "_" "-")
JOB_DIR="gs://$PROJECT-jobs/$JOB_PATH/"

PACKAGES="$HOME/Documents/trfl/dist/trfl-1.0.tar.gz,$HOME/Documents/pyoneer/dist/pyoneer-0.0.0.tar.gz,$HOME/Documents/box2d-py/dist/box2d_py-2.3.8-cp35-cp35m-linux_x86_64.whl"

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket "gs://tool-use-staging" \
    --package-path "$(pwd)/tool_use" \
    --job-dir $JOB_DIR \
    --packages $PACKAGES \
    --module-name "tool_use.main" \
    --scale-tier "basic" \
    --runtime-version "1.12" \
    --python-version "3.5" \
    --region "us-west1" \
    --stream-logs \
    -- \
    --env-name $ENV_NAME \
    --seed $SEED