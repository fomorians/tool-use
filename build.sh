#!/usr/bin/env bash

set -e

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME="tool_use"
IMAGE_TAG="tool_use"
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG"

PREFIX="$HOME/Documents"
PACKAGES_DIR="packages/"
PYCOLAB_PACKAGE="$PREFIX/pycolab/dist/pycolab-1.2.tar.gz"
GYM_PYCOLAB_PACKAGE="$PREFIX/gym_pycolab/dist/gym-pycolab-1.0.0.tar.gz"
GYM_TOOL_USE_PACKAGE="$PREFIX/gym_tool_use/dist/gym-tool-use-1.0.0.tar.gz"
PYONEER_PACKAGE="$PREFIX/pyoneer/dist/fomoro-pyoneer-0.3.0.tar.gz"

cp $PYCOLAB_PACKAGE $PACKAGES_DIR
cp $GYM_PYCOLAB_PACKAGE $PACKAGES_DIR
cp $GYM_TOOL_USE_PACKAGE $PACKAGES_DIR
cp $PYONEER_PACKAGE $PACKAGES_DIR

docker build -f Dockerfile-gpu -t $IMAGE_URI .
# docker run --runtime=nvidia $IMAGE_URI --epochs 1
# docker push $IMAGE_URI
