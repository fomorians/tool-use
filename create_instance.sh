#!/usr/bin/env bash

INSTANCE_NAME=$1
MACHINE_TYPE=n1-standard-8
ZONE=us-west1-b
SCOPES=default,storage-rw

IMAGE_FAMILY=tf-latest-gpu
IMAGE_PROJECT=deeplearning-platform-release

# BOOT_DISK_NAME=$1
BOOT_DISK_TYPE=pd-ssd
BOOT_DISK_SIZE=30GB

GPU_TYPE=nvidia-tesla-k80
GPU_COUNT=1

# --disk="name=$BOOT_DISK_NAME,device-name=$BOOT_DISK_NAME,mode=rw,boot=yes" \

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-type=$BOOT_DISK_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --no-boot-disk-auto-delete \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --scopes=$SCOPES \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --metadata="install-nvidia-driver=True"
