#!/usr/bin/python3
"""
RGB-driven variant of run_competition.py: instead of subscribing to Flightmare's
ground-truth depth topic, this subscribes to the RGB camera topic, runs it through
Depth Anything V2 (metric, VKITTI, ViT-S -- see models/DepthAnythingV2/) to get a
monocular depth estimate, converts that into the same normalized encoding the
ViTLSTM policy was trained on (via the empirical calibration in
models/DepthAnythingV2/depth_calibration.npz), and feeds that to the *same*
compute_command_vision_based used by the ground-truth-depth pipeline.

This is an experimental substitution -- expect it to fly noticeably worse than
the ground-truth-depth pipeline, since DA2 was never trained on Flightmare's
synthetic renders and the meters->normalized-target calibration is an empirical
approximation (see models/DepthAnythingV2/calibrate.py).
"""
import argparse

import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

from envsim_msgs.msg import ObstacleArray

from user_code import compute_command_vision_based
from utils import AgileCommandMode, AgileQuadState

import time
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
from copy import deepcopy
import cv2
import torch

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models/DepthAnythingV2'))
from depth_estimator import DepthEstimator


class AgilePilotNodeRGB:
    def __init__(self, model_type=None, model_path=None, desVel=None):
        print("[RUN_COMPETITION_RGB] Initializing agile_pilot_node_rgb...")
        rospy.init_node("agile_pilot_node_rgb", anonymous=False)

        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None

        quad_name = "kingfisher"

        self.ctr = 0
        self.t1 = 0
        self.last_valid_img = None
        data_log_format = {'timestamp': [], 'desired_vel': [],
                            'quat_1': [], 'quat_2': [], 'quat_3': [], 'quat_4': [],
                            'pos_x': [], 'pos_y': [], 'pos_z': [],
                            'vel_x': [], 'vel_y': [], 'vel_z': [],
                            'velcmd_x': [], 'velcmd_y': [], 'velcmd_z': [],
                            'ct_cmd': [], 'br_cmd_x': [], 'br_cmd_y': [], 'br_cmd_z': [],
                            'is_collide': []}
        self.data_log = pd.DataFrame(data_log_format)
        self.count = 0
        self.time_interval = .03

        self.folder = f"train_set_rgb/{int(time.time() * 100)}"
        os.makedirs(self.folder, exist_ok=True)

        self.desiredVel = desVel
        print(f"\n[RUN_COMPETITION_RGB] Desired velocity = {self.desiredVel}\n")

        print(f"[RUN_COMPETITION_RGB] Loading depth estimator (Depth Anything V2, metric VKITTI, ViT-S)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_estimator = DepthEstimator(device=self.device)
        print(f"[RUN_COMPETITION_RGB] Depth estimator loaded")

        print(f"[RUN_COMPETITION_RGB] Model loading from {model_path} ...")
        if model_type == 'LSTMNet':
            self.model = LSTMNet().to(self.device).float()
        elif model_type == 'UNetLSTM':
            self.model = UNetConvLSTMNet().to(self.device).float()
        elif model_type == 'ConvNet':
            self.model = ConvNet().to(self.device).float()
        elif model_type == 'ViT':
            self.model = ViT().to(self.device).float()
        elif model_type == 'ViTLSTM':
            self.model = LSTMNetVIT().to(self.device).float()
        else:
            print(f'[RUN_COMPETITION_RGB] Invalid model_type {model_type}. Exiting.')
            exit()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model_hidden_state = None
        print(f"[RUN_COMPETITION_RGB] Model loaded")
        time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0
        self.curr_cmd = None

        self.start_sub = rospy.Subscriber(
            "/" + quad_name + "/start_navigation", Empty, self.start_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.odom_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/state", QuadState, self.state_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.rgb_img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/image", Image, self.rgb_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles", ObstacleArray, self.obstacle_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.cmd_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/command", Command, self.cmd_callback,
            queue_size=1, tcp_nodelay=True,
        )

        self.cmd_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/feedthrough_command", Command, queue_size=1,
        )
        self.linvel_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/velocity_command", TwistStamped, queue_size=1,
        )
        self.debug_img1_pub = rospy.Publisher("/debug_img1", Image, queue_size=1)
        self.debug_img2_pub = rospy.Publisher("/debug_img2", Image, queue_size=1)

        self.col = None
        print("[RUN_COMPETITION_RGB] Initialization completed!")

    def cmd_callback(self, msg):
        self.curr_cmd = msg

    def obstacle_callback(self, obs_data):
        if self.state is None or not obs_data.obstacles:
            return
        dist = np.linalg.norm([obs_data.obstacles[0].position.x,
                                obs_data.obstacles[0].position.y,
                                obs_data.obstacles[0].position.z])
        margin = dist - obs_data.obstacles[0].scale
        self.col = 1 if (margin < 0 or self.state.pos[2] <= 0.01) else 0

    def rgb_callback(self, img_data):
        self.ctr += 1
        rgb = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        # the unity/image topic is bgr8, so a passthrough conversion is already in
        # OpenCV's BGR order -- which is what DA2's infer_image expects. Flipping here
        # would hand it red/blue swapped.
        bgr = rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)

        img = self.depth_estimator.predict_normalized(bgr.copy())
        prev_img = deepcopy(self.last_valid_img) if self.last_valid_img is not None else img
        self.last_valid_img = deepcopy(img)

        if self.state is None:
            return

        start_compute_time = time.time()
        command, (debug_img1, debug_img2), self.model_hidden_state = compute_command_vision_based(
            self.state, img, prev_img, self.desiredVel, self.model, self.model_hidden_state)

        self.debug_img1_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="passthrough"))
        self.debug_img2_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="passthrough"))

        if self.ctr % 30 == 0:
            print(f'[RUN_COMPETITION_RGB] compute_command_vision_based (via DA2) took '
                  f'{time.time() - start_compute_time:.4f} seconds')

        self.publish_command(command)

        if self.state.pos[0] < 0.1:
            self.start_time = command.t
        if self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            with open("timeTaken_rgb.dat", "a") as f:
                f.write(str(float(command.t - self.start_time)) + "\n")
            self.logged_time_flag = 1

        if (self.state.t - self.t1 > self.time_interval or self.t1 == 0) and self.state.pos[0] < 63:
            self.t1 = self.state.t
            timestamp = round(self.state.t, 3)
            cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img * 255).astype(np.uint8))
            col = self.col if self.col is not None else 0
            self.data_log.loc[len(self.data_log)] = [
                timestamp, self.desiredVel,
                self.state.att[0], self.state.att[1], self.state.att[2], self.state.att[3],
                self.state.pos[0], self.state.pos[1], self.state.pos[2],
                self.state.vel[0], self.state.vel[1], self.state.vel[2],
                command.velocity[0], command.velocity[1], command.velocity[2],
                self.curr_cmd.collective_thrust if self.curr_cmd else 0.0,
                self.curr_cmd.bodyrates.x if self.curr_cmd else 0.0,
                self.curr_cmd.bodyrates.y if self.curr_cmd else 0.0,
                self.curr_cmd.bodyrates.z if self.curr_cmd else 0.0,
                col,
            ]
            self.count += 1

        if self.count % 5 == 0:
            self.data_log.to_csv(self.folder + "/data.csv")

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)

    def publish_command(self, command):
        if command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
        else:
            assert False, "Unknown/unsupported command mode for run_competition_rgb"

    def start_callback(self, data):
        print("[RUN_COMPETITION_RGB] Start publishing commands!")
        self.publish_commands = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot (RGB -> Depth Anything V2 -> policy).")
    parser.add_argument('--model_type', type=str, default='ViTLSTM')
    parser.add_argument('--model_path', type=str, default='../../models/ViTLSTM_model.pth')
    parser.add_argument('--des_vel', type=float, default=None)
    args = parser.parse_args()
    node = AgilePilotNodeRGB(model_type=args.model_type, model_path=args.model_path, desVel=args.des_vel)
    rospy.spin()
