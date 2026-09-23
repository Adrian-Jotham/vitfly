#!/usr/bin/env python3
"""
Offline version of calibrate.py: fits the DA2-meters -> policy-normalized-depth
lookup table from already-collected paired frames on disk instead of from a live
simulator.

Needs a train_set collected in state/datagen mode (`bash launch_evaluation.bash N state`),
which writes both `<timestamp>.png` (Flightmare's ground-truth depth, already in the
policy's normalized encoding scaled to 0-255) and `<timestamp>_rgb.png` for each frame.
Those pairs are the same signal calibrate.py samples live, so the fit is equivalent --
but reproducible, faster, and it covers whatever range the expert actually flew through.

Usage:
    python3 calibrate_offline.py --in envtest/ros/train_set [--out depth_calibration.npz]
"""

import argparse
import glob
import os
import sys
import time
from os.path import join as opj

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_estimator import DepthEstimator

PIXEL_STRIDE = 6
N_BINS = 60
MIN_SAMPLES_PER_BIN = 50


def main():
    parser = argparse.ArgumentParser(description="Fit the DA2 depth calibration from collected frame pairs.")
    parser.add_argument("--in", dest="in_dir", required=True, help="collected train_set directory")
    parser.add_argument("--out", dest="out_path",
                        default=opj(os.path.dirname(os.path.abspath(__file__)), "depth_calibration.npz"))
    parser.add_argument("--frame-stride", type=int, default=1, help="use every Nth frame")
    args = parser.parse_args()

    pairs = []
    for traj_dir in sorted(glob.glob(opj(args.in_dir, "*"))):
        if not os.path.isdir(traj_dir):
            continue
        for rgb_file in sorted(glob.glob(opj(traj_dir, "*_rgb.png"))):
            gt_file = rgb_file[: -len("_rgb.png")] + ".png"
            if os.path.exists(gt_file):
                pairs.append((rgb_file, gt_file))
    pairs = pairs[:: args.frame_stride]

    if len(pairs) < 20:
        print(f"[calibrate_offline] Only {len(pairs)} rgb/depth pairs found in {args.in_dir}. "
              f"Collect some with `bash launch_evaluation.bash N state` first.")
        sys.exit(1)

    print(f"[calibrate_offline] Found {len(pairs)} paired frames. Loading Depth Anything V2 ...")
    estimator = DepthEstimator()

    meters_all, target_all = [], []
    start = time.time()
    for i, (rgb_file, gt_file) in enumerate(pairs):
        # unity/image is bgr8 -> what was written to disk is already BGR, which is
        # what DA2 expects. No channel flip.
        bgr = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue

        meters = estimator.predict(bgr)
        target = gt.astype(np.float32) / 255.0
        if meters.shape != target.shape:
            target = cv2.resize(target, (meters.shape[1], meters.shape[0]))

        meters_all.append(meters[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel())
        target_all.append(target[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel())

        if (i + 1) % 100 == 0:
            print(f"[calibrate_offline] {i+1}/{len(pairs)} frames ({time.time()-start:.0f}s)")

    meters = np.concatenate(meters_all)
    target = np.concatenate(target_all)
    print(f"[calibrate_offline] {len(meters)} pixel samples, DA2 range "
          f"{meters.min():.2f}-{meters.max():.2f}m")

    bin_edges = np.linspace(0, min(meters.max(), 80.0), N_BINS + 1)
    centers, targets = [], []
    for i in range(N_BINS):
        mask = (meters >= bin_edges[i]) & (meters < bin_edges[i + 1])
        if mask.sum() < MIN_SAMPLES_PER_BIN:
            continue
        centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        targets.append(np.median(target[mask]))

    centers = np.array(centers)
    # Far/background pixels clip at target=1.0, so target is monotonically
    # non-decreasing with distance (closer -> lower -> more "danger" signal).
    targets = np.maximum.accumulate(np.array(targets))

    np.savez(args.out_path, meters=centers, target=targets,
             n_frames=len(pairs), n_samples=len(meters))
    print(f"[calibrate_offline] Saved {len(centers)}-point curve to {args.out_path}")
    print(f"[calibrate_offline] Covered {centers.min():.2f}-{centers.max():.2f}m; "
          f"target {targets.min():.3f} -> {targets.max():.3f}")


if __name__ == "__main__":
    main()
