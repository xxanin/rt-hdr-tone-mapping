#!/usr/bin/env bash
set -e

echo "[LOG] Cleaning up old output"
dx rm -rf final_output.mp4

echo "[LOG] Running pipeline"
dx env LD_LIBRARY_PATH=./remote_libs ./build_remote/cuda-webcam-filter \
  --input video \
  --path test2.mp4 \
  --filter hdr \
  --exposure 2.0 \
  --gamma 2.2 \
  --saturation 1.5 \
  --tone-algo 1 \
  --target-filter emboss \
  --transition-duration 5000 \
  --streams 4

echo "[LOG] Video saved as final_output.mp4"
