#!/usr/bin/env python3
"""
Builds a "pseudo-depth" training dataset by running Depth Anything V2 over the
RGB frames of a collected trajectory set, so a policy can be trained on the same
depth distribution it will actually see at test time under the RGB pipeline
(envtest/ros/run_competition_rgb.py) instead of on Flightmare's ground-truth depth.

Input:  a collected train_set directory, i.e. folders of
            <timestamp>.png       ground-truth depth   (written by run_competition.py)
            <timestamp>_rgb.png   rgb frame            (same, state/datagen mode only)
            data.csv              telemetry
Output: a dataset directory in the exact layout training/dataloading.py expects
            <timestamp>.png       DA2 pseudo-depth, uint8 480x640, value/255 = normalized depth
            data.csv              copied verbatim

Usage:
    python3 make_da2_dataset.py --in envtest/ros/train_set --out training/datasets/data_da2
"""

import argparse
import glob
import os
import shutil
import sys
import time
from os.path import join as opj

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_estimator import DepthEstimator


def convert_trajectory(traj_dir, out_dir, estimator, compare=False):
    rgb_files = sorted(glob.glob(opj(traj_dir, "*_rgb.png")))
    if not rgb_files:
        return 0, None

    os.makedirs(out_dir, exist_ok=True)
    errors = []

    for rgb_file in rgb_files:
        # the unity/image topic is bgr8, so run_competition.py wrote an already-BGR
        # array through cv2.imwrite (correct on disk), and imread hands it back as
        # BGR -- exactly what DA2's infer_image expects, so no channel flip.
        bgr = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        normalized = estimator.predict_normalized(bgr)

        timestamp = os.path.basename(rgb_file)[: -len("_rgb.png")]
        cv2.imwrite(opj(out_dir, f"{timestamp}.png"), (normalized * 255).astype(np.uint8))

        if compare:
            gt_file = opj(traj_dir, f"{timestamp}.png")
            if os.path.exists(gt_file):
                gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                if gt.shape == normalized.shape:
                    errors.append(float(np.abs(gt - normalized).mean()))

    csv_src = opj(traj_dir, "data.csv")
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, opj(out_dir, "data.csv"))

    mean_err = float(np.mean(errors)) if errors else None
    return len(rgb_files), mean_err


def main():
    parser = argparse.ArgumentParser(description="Build a DA2 pseudo-depth dataset from collected RGB frames.")
    parser.add_argument("--in", dest="in_dir", required=True, help="collected train_set directory")
    parser.add_argument("--out", dest="out_dir", required=True, help="output dataset directory")
    parser.add_argument("--compare", action="store_true",
                        help="also report mean abs error vs the paired ground-truth depth frames")
    args = parser.parse_args()

    traj_dirs = sorted([d for d in glob.glob(opj(args.in_dir, "*")) if os.path.isdir(d)])
    if not traj_dirs:
        print(f"[make_da2_dataset] No trajectory folders found in {args.in_dir}")
        sys.exit(1)

    print(f"[make_da2_dataset] Loading Depth Anything V2 ...")
    estimator = DepthEstimator()

    total_frames = 0
    all_errors = []
    skipped = []
    start = time.time()

    for i, traj_dir in enumerate(traj_dirs):
        name = os.path.basename(traj_dir)
        n, err = convert_trajectory(traj_dir, opj(args.out_dir, name), estimator, compare=args.compare)
        if n == 0:
            skipped.append(name)
            continue
        total_frames += n
        if err is not None:
            all_errors.append(err)
        print(f"[make_da2_dataset] [{i+1}/{len(traj_dirs)}] {name}: {n} frames"
              + (f", mean abs err vs GT depth {err:.4f}" if err is not None else ""))

    print(f"\n[make_da2_dataset] Done: {total_frames} frames across "
          f"{len(traj_dirs) - len(skipped)} trajectories in {time.time()-start:.1f}s")
    if skipped:
        print(f"[make_da2_dataset] Skipped {len(skipped)} folder(s) with no *_rgb.png "
              f"(collect with `launch_evaluation.bash N state`): {skipped[:5]}")
    if all_errors:
        print(f"[make_da2_dataset] Mean abs error vs ground-truth depth, over trajectories: "
              f"{np.mean(all_errors):.4f} (normalized [0,1] units)")
    print(f"[make_da2_dataset] Train on it with `dataset = {os.path.basename(args.out_dir)}` "
          f"in training/config/train.txt")


if __name__ == "__main__":
    main()
