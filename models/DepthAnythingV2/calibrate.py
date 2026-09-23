#!/usr/bin/env python3
"""
Empirically calibrates a mapping from Depth Anything V2's predicted metric
depth (meters) to the normalized encoding envtest/ros/run_competition.py's
img_callback feeds the ViTLSTM policy: clip(raw_unity_depth / 0.09, 0, 1).

We don't know Flightmare's exact depth-buffer encoding formula (no Unity C#
source in this repo), so instead of guessing it analytically, this samples
live (RGB, raw depth) frame pairs from the running simulator -- both views of
the *same* scene at the *same* instant -- runs DA2 on the RGB frame, and fits
a monotonic per-pixel mapping from DA2's meters to the policy's normalized
target. This also naturally absorbs DA2's own domain-gap bias on Unity's
synthetic renders, which a meters-based analytical conversion would not.

Run this against a live simulator (e.g. while `launch_evaluation.bash N
vision` is running in another shell, so obstacles are actually in frame at
varying distances) and it writes depth_calibration.npz next to this file.
"""

import os
import sys
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from envsim_msgs.msg import ObstacleArray

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_estimator import DepthEstimator

QUAD_NAME = "kingfisher"
DEPTH_IM_THRESHOLD = 0.09  # matches envtest/ros/run_competition.py
PIXEL_STRIDE = 4  # subsample every Nth pixel per frame to keep the fit dataset small
MAX_WAIT_S = 240

# A policy that's actively avoiding obstacles spends most of its time far from
# anything (raw_depth/0.09 clipped at 1, no useful signal), so a plain random
# sample of frames badly under-covers the near-field/imminent-collision regime
# that matters most. Gate extra sampling on ground-truth obstacle proximity
# (from /.../groundtruth/obstacles) to specifically catch those close calls.
N_FAR_FRAMES = 15
N_CLOSE_FRAMES = 30
CLOSE_DIST_M = 4.0
MIN_FRAME_GAP_S = 0.3  # avoid re-sampling the same close encounter every callback tick


class Calibrator:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_rgb_t = None
        self.latest_depth = None
        self.latest_depth_t = None
        self.samples_meters = []
        self.samples_target = []
        self.n_far = 0
        self.n_close = 0
        self.nearest_obstacle_m = None
        self.last_sample_t = 0.0

        self.destimator = DepthEstimator()

        rospy.Subscriber(f"/{QUAD_NAME}/dodgeros_pilot/unity/image", Image, self._rgb_cb, queue_size=1)
        rospy.Subscriber(f"/{QUAD_NAME}/dodgeros_pilot/unity/depth", Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(f"/{QUAD_NAME}/dodgeros_pilot/groundtruth/obstacles", ObstacleArray,
                          self._obstacle_cb, queue_size=1)

    def _obstacle_cb(self, msg):
        if not msg.obstacles:
            return
        dists = [
            np.linalg.norm([o.position.x, o.position.y, o.position.z]) - o.scale
            for o in msg.obstacles
        ]
        self.nearest_obstacle_m = min(dists)

    def _rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_rgb_t = msg.header.stamp.to_sec()
        self._maybe_sample()

    def _depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_depth_t = msg.header.stamp.to_sec()

    def _maybe_sample(self):
        self._ticks = getattr(self, "_ticks", 0) + 1
        if self.n_far >= N_FAR_FRAMES and self.n_close >= N_CLOSE_FRAMES:
            return
        if self.latest_rgb is None or self.latest_depth is None:
            return
        gap = abs(self.latest_rgb_t - self.latest_depth_t)
        if self._ticks % 50 == 0:
            print(f"[calibrate] [debug] tick={self._ticks} rgb_t={self.latest_rgb_t:.3f} "
                  f"depth_t={self.latest_depth_t:.3f} gap={gap:.3f} "
                  f"since_last_sample={self.latest_rgb_t - self.last_sample_t:.3f} "
                  f"nearest_obs={self.nearest_obstacle_m}")
        if gap > 0.5:
            return  # frames not close enough in time
        if self.latest_rgb_t - self.last_sample_t < MIN_FRAME_GAP_S:
            return

        is_close = self.nearest_obstacle_m is not None and self.nearest_obstacle_m < CLOSE_DIST_M
        if is_close and self.n_close >= N_CLOSE_FRAMES:
            return
        if not is_close and self.n_far >= N_FAR_FRAMES:
            return

        self.last_sample_t = self.latest_rgb_t
        rgb = self.latest_rgb
        raw_depth = self.latest_depth
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        # unity/image is bgr8, so passthrough is already BGR -- what DA2 expects. No flip.
        bgr = rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)

        meters = self.destimator.predict(bgr.copy())
        target = np.clip(raw_depth / DEPTH_IM_THRESHOLD, 0, 1)

        if meters.shape != target.shape:
            # depth topic and rgb topic can differ in resolution; resize target to match
            import cv2
            target = cv2.resize(target, (meters.shape[1], meters.shape[0]))

        m = meters[::PIXEL_STRIDE, ::PIXEL_STRIDE].flatten()
        t = target[::PIXEL_STRIDE, ::PIXEL_STRIDE].flatten()
        self.samples_meters.append(m)
        self.samples_target.append(t)
        if is_close:
            self.n_close += 1
        else:
            self.n_far += 1
        tag = "CLOSE" if is_close else "far"
        nearest = f"{self.nearest_obstacle_m:.2f}m" if self.nearest_obstacle_m is not None else "?"
        print(f"[calibrate] [{tag}] far={self.n_far}/{N_FAR_FRAMES} close={self.n_close}/{N_CLOSE_FRAMES} "
              f"(nearest obstacle {nearest}, DA2 range {m.min():.2f}-{m.max():.2f}m)")

    @property
    def frames_used(self):
        return self.n_far + self.n_close

    def done(self):
        return self.n_far >= N_FAR_FRAMES and self.n_close >= N_CLOSE_FRAMES

    def fit_and_save(self, out_path):
        meters = np.concatenate(self.samples_meters)
        target = np.concatenate(self.samples_target)

        # bin by predicted meters, take the median target per bin, monotonic by construction
        # of the underlying physical relationship (closer -> higher target value)
        n_bins = 60
        max_m = min(meters.max(), 80.0)
        bin_edges = np.linspace(0, max_m, n_bins + 1)
        bin_centers = []
        bin_targets = []
        for i in range(n_bins):
            mask = (meters >= bin_edges[i]) & (meters < bin_edges[i + 1])
            if mask.sum() < 20:
                continue
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_targets.append(np.median(target[mask]))

        bin_centers = np.array(bin_centers)
        bin_targets = np.array(bin_targets)

        # Far/background samples clip at target=1.0 (raw_depth >= threshold), so target
        # must be monotonically non-decreasing with distance (closer -> lower target,
        # i.e. more "danger" signal). Enforce that via a running max from the near end
        # to smooth out per-bin noise while preserving this direction.
        order = np.argsort(bin_centers)
        bin_centers = bin_centers[order]
        bin_targets = bin_targets[order]
        bin_targets = np.maximum.accumulate(bin_targets)

        np.savez(out_path, meters=bin_centers, target=bin_targets,
                 n_frames=self.frames_used, n_samples=len(meters))
        print(f"[calibrate] saved {len(bin_centers)}-point calibration curve to {out_path}")
        print(f"[calibrate] meters range covered: {bin_centers.min():.2f} - {bin_centers.max():.2f}")


if __name__ == "__main__":
    rospy.init_node("da2_calibrate", anonymous=True)
    cal = Calibrator()
    start = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and not cal.done() and (time.time() - start) < MAX_WAIT_S:
        rate.sleep()

    if cal.frames_used < 10:
        print(f"[calibrate] ERROR: only collected {cal.frames_used} frames, need at least 10. "
              f"Is the simulator actually running and moving through a scene?")
        sys.exit(1)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_calibration.npz")
    cal.fit_and_save(out_path)
