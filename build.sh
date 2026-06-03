#!/usr/bin/env bash
set -e

echo "[LOG] Building inside cached cv-builder container"
sudo docker run --rm -v "$PWD":/workspace -w /workspace cv-builder cmake --build build_remote -j$(nproc)
echo "[LOG] Build complete"
