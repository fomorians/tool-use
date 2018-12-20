#!/usr/bin/env bash
TIMESTAMP=$(date +"%s")
ENV_NAME="BipedalWalker-v2"
ENV_NAME_LOWER="bipedalwalker-v2"
ENV_NAME_SAFE="bipedalwalker_v2"
SEED="$TIMESTAMP"
BUCKET_NAME="tool-use"
JOB_NAME="${ENV_NAME_SAFE}_${SEED}"
JOB_DIR="gs://$BUCKET_NAME/jobs/$ENV_NAME_LOWER/$SEED/"
SCALE_TIER="basic"
PACKAGE_PATH="$HOME/Documents/tool_use/"
MODULE_NAME="tool_use.main"
RUNTIME_VERSION="1.12"
PYTHON_VERSION="3.5"
STAGING_BUCKET="gs://$BUCKET_NAME"
REGION="us-west1"
PACKAGES="$HOME/Documents/pyoneer/dist/pyoneer-0.0.0.tar.gz"

gcloud ml-engine jobs submit training $JOB_NAME \
        --staging-bucket $STAGING_BUCKET \
        --package-path $PACKAGE_PATH \
        --module-name $MODULE_NAME \
        --job-dir $JOB_DIR \
        --region $REGION \
        --scale-tier $SCALE_TIER \
        --python-version $PYTHON_VERSION \
        --packages $PACKAGES \
        -- \
        --job-dir $JOB_DIR \
        --env-name $ENV_NAME \
        --seed $SEED