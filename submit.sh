#!/usr/bin/env bash
ENV=$1
SEED=$2
DESC=$3
PROJECT="tool-use"

JOB_NAME=$(echo "${ENV}_${SEED}_${DESC}" | tr "-" "_")
JOB_DIR="gs://$PROJECT-jobs/$ENV/$SEED/$DESC"

PACKAGES="$HOME/Documents/pyoneer/dist/pyoneer-0.0.0.tar.gz"

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket "gs://tool-use-staging" \
    --package-path "$(pwd)/tool_use" \
    --job-dir $JOB_DIR \
    --packages $PACKAGES \
    --module-name "tool_use.main" \
    --config "$(pwd)/config.yaml" \
    -- \
    --env $ENV \
    --seed $SEED