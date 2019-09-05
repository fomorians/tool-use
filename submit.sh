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

PREFIX="$HOME/Documents"
PYCOLAB_PACKAGE="$PREFIX/pycolab/dist/pycolab-1.2.tar.gz"
GYM_PYCOLAB_PACKAGE="$PREFIX/gym_pycolab/dist/gym-pycolab-1.0.0.tar.gz"
GYM_TOOL_USE_PACKAGE="$PREFIX/gym_tool_use/dist/gym-tool-use-1.0.0.tar.gz"
PYONEER_PACKAGE="$PREFIX/pyoneer/dist/fomoro-pyoneer-0.3.0.tar.gz"
PACKAGES="$PYCOLAB_PACKAGE,$GYM_PYCOLAB_PACKAGE,$GYM_TOOL_USE_PACKAGE,$PYONEER_PACKAGE"

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket "gs://tool-use-staging" \
    --package-path "$(pwd)/tool_use" \
    --job-dir ${JOB_DIR} \
    --packages ${PACKAGES} \
    --module-name "tool_use.sac.train" \
    --config "$(pwd)/config.yaml" \
    --stream-logs \
    -- \
    --env ${ENV_NAME} \
    --seed ${SEED} \
    $@
