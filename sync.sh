#!/usr/bin/env bash
INSTANCE_NAME=$1
gcloud compute scp --recurse . $INSTANCE_NAME:~/tool_use
gcloud compute scp --recurse ~/Documents/pyoneer $INSTANCE_NAME:~/pyoneer
gcloud compute scp --recurse $INSTANCE_NAME:~/jobs/ ~/jobs/tool_use/
