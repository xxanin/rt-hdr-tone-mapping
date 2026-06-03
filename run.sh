#!/usr/bin/env bash
set -e

echo "[LOG] Cleaning up old output"
dx rm -rf final_output.mp4

echo "[LOG] Executing remote processing..."
dx env LD_LIBRARY_PATH=./remote_libs ./build_remote/cuda-webcam-filter --filter hdr --exposure 1.5 --gamma 2.2 --saturation 1.2 --tone-algo 1

echo "[LOG] Video saved as final_output.mp4"
