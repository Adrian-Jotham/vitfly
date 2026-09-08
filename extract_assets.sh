#!/bin/bash
# Extracts the Datashare tar/zip assets from downloads/ into the locations
# the upstream README specifies. Run this on the host (no Docker/ROS needed) after
# manually downloading files from https://upenn.app.box.com/v/ViT-quad-datashare
# (pw: vitfly2025) into downloads/.
set -euo pipefail
cd "$(dirname "$0")"

DL=downloads
found=0

if [ -f "$DL/environments.tar" ]; then
  echo "Extracting environments.tar -> flightmare/flightpy/configs/vision"
  tar -xvf "$DL/environments.tar" -C flightmare/flightpy/configs/vision
  found=1
fi

if [ -f "$DL/flightrender.tar" ]; then
  echo "Extracting flightrender.tar -> flightmare/flightrender"
  tar -xvf "$DL/flightrender.tar" -C flightmare/flightrender
  found=1
fi

if [ -f "$DL/pretrained_models.tar" ]; then
  echo "Extracting pretrained_models.tar -> models"
  tar -xvf "$DL/pretrained_models.tar" -C models
  found=1
fi

if [ -f "$DL/data.zip" ]; then
  echo "Extracting data.zip -> training/datasets/data (this may take a while)"
  mkdir -p training/datasets/data training/logs
  unzip -q "$DL/data.zip" -d training/datasets/data
  found=1
fi

if [ "$found" -eq 0 ]; then
  echo "No known asset files found in $DL/. Expected one or more of:"
  echo "  environments.tar  flightrender.tar  pretrained_models.tar  data.zip"
  exit 1
fi

echo "Done."
