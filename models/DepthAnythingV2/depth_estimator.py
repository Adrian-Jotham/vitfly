"""
Thin wrapper around Depth Anything V2 (metric, outdoor/VKITTI, ViT-S) for use
as a drop-in monocular depth source ahead of the ViTLSTM obstacle-avoidance
policy in models/model.py.

Usage:
    from depth_estimator import DepthEstimator
    de = DepthEstimator(device=torch.device("cuda"))
    meters = de.predict(bgr_uint8_image)   # HxW float32, meters
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_anything_v2.dpt import DepthAnythingV2

_MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
_MAX_DEPTH_METERS = 80.0  # VKITTI (outdoor) metric checkpoints are trained with max_depth=80
_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "depth_anything_v2_metric_vkitti_vits.pth",
)
_CALIBRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_calibration.npz")


class DepthEstimator:
    def __init__(self, device=None, checkpoint_path=_CHECKPOINT, input_size=518,
                 calibration_path=_CALIBRATION):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.model = DepthAnythingV2(**_MODEL_CONFIG, max_depth=_MAX_DEPTH_METERS)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()

        self._cal_meters = None
        self._cal_target = None
        if calibration_path and os.path.exists(calibration_path):
            cal = np.load(calibration_path)
            self._cal_meters = cal["meters"]
            self._cal_target = cal["target"]

    @torch.no_grad()
    def predict(self, bgr_image):
        """bgr_image: HxWx3 uint8, OpenCV BGR convention. Returns HxW float32 depth in meters."""
        return self.model.infer_image(bgr_image, input_size=self.input_size)

    def predict_normalized(self, bgr_image):
        """
        Returns an HxW float32 array in [0,1] in the same convention the ViTLSTM
        policy was trained on (see envtest/ros/run_competition.py's img_callback:
        clip(raw_unity_depth / 0.09, 0, 1)), via the empirical calibration curve
        fit by calibrate.py against the live simulator's own ground-truth depth.
        Values closer than the calibration's covered range flat-extrapolate to
        its nearest fitted value (see calibrate.py docstring for the caveat).
        """
        if self._cal_meters is None:
            raise RuntimeError(
                "No depth_calibration.npz found next to depth_estimator.py -- "
                "run calibrate.py against a live simulator first."
            )
        meters = self.predict(bgr_image)
        return np.interp(meters, self._cal_meters, self._cal_target).astype(np.float32)
