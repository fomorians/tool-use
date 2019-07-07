#!/usr/bin/env bash
ENV_NAME="$1"
SEED="$2"
DESC="$3"

# shift position parameters to the left 3 to pass as argparse arguments to tool_use.main
shift 3

NOW=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=$(echo "${ENV_NAME}_${SEED}_${DESC}_${NOW}" | tr "-" "_")
JOB_DIR="gs://tool-use-jobs/$ENV_NAME/$SEED/$DESC"

PREFIX="$HOME/code"
PYCOLAB_PACKAGE="$PREFIX/pycolab/dist/pycolab-1.2.tar.gz"
GYM_PYCOLAB_PACKAGE="$PREFIX/gym_pycolab/dist/gym_pycolab-0.0.0.tar.gz"
GYM_TOOL_USE_PACKAGE="$PREFIX/gym_tool_use/dist/gym_tool_use-1.0.0.tar.gz"
PYONEER_PACKAGE="$PREFIX/pyoneer/dist/pyoneer-0.0.0.tar.gz"
PACKAGES="$PYCOLAB_PACKAGE,$GYM_PYCOLAB_PACKAGE,$GYM_TOOL_USE_PACKAGE,$PYONEER_PACKAGE"

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket "gs://tool-use-staging" \
    --package-path "$(pwd)/tool_use" \
    --job-dir ${JOB_DIR} \
    --packages ${PACKAGES} \
    --module-name "tool_use.main" \
    --config "$(pwd)/config.yaml" \
    --stream-logs \
    -- \
    --env-name ${ENV_NAME} \
    --seed ${SEED} \
    $@
