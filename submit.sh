#!/usr/bin/env bash

set -e

ENV_NAME="$1"
SEED="$2"
DESC="$3"

# shift position parameters to the left 3 to pass as argparse arguments to tool_use.main
shift 3

NOW=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=$(echo "${ENV_NAME}_${SEED}_${DESC}_${NOW}" | tr "-" "_")
JOB_DIR="gs://tool-use-jobs/$ENV_NAME/$SEED/$DESC"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME="tool_use"
IMAGE_TAG="tool_use"
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG"

REGION="us-west1"
SCALE_TIER="CUSTOM"
MACHINE_TYPE="standard_gpu"

gcloud ai-platform jobs submit training $JOB_NAME \
    --scale-tier $SCALE_TIER \
    --master-machine-type $MACHINE_TYPE \
    --master-image-uri $IMAGE_URI \
    --stream-logs \
    --region $REGION \
    -- \
    --job-dir $JOB_DIR \
    --seed $SEED \
    --env $ENV_NAME \
    $@
