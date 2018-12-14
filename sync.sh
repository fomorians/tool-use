#!/usr/bin/env bash
gcloud compute scp --recurse . tool-use:~/tool_use
gcloud compute scp --recurse ~/Documents/pyoneer tool-use:~/pyoneer
gcloud compute scp --recurse tool-use:~/jobs ~/jobs/tool_use/
