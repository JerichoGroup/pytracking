import rospy
import select
import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import torch

from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation import Tracker
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from tracker.optical_flow import VisualTrackerKLT


class ObjectTracker:
    def __init__(self):
        self.clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = rospy.get_param("tracker/ip", "172.17.0.1")
        self.port = int(rospy.get_param("tracker/port", 5556))
        try:
            self.clientsocket.connect((self.ip, self.port))
        except Exception as e:
            self.clientsocket = None
        self.run_optical_flow = bool(
            int(rospy.get_param("tracker/run_optical_flow", 0))
        )
        self.tracker_run_iter = int(rospy.get_param("tracker/track_run_iter", 10))
        self.init_timeout = int(rospy.get_param("tracker/init_timeout", 20))
        self.tracker_counter = 0
        rospy.init_node("tracker")
        rospy.Subscriber("image", Image, self.image_callback, queue_size=1)
        self.bb_pub = rospy.Publisher("bounding_box", Float32MultiArray, queue_size=10)
        self.tracker = Tracker("dimp", "dimp18")
        self.init = False
        self.of = VisualTrackerKLT()
        self.bridge = CvBridge()

    def image_callback(self, image_msg):
        if self.clientsocket is None:
            self.reconnect_to_socket()
        if self.tracker_counter > 100:
            self.tracker_counter = 0
        self.tracker_counter += 1
        modify_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        orig = modify_image.copy()
        if not self.init:
            if self.clientsocket is None:
                return
            bounding_box = self.init_bounding_box(orig)
            if self.run_optical_flow:
                self.of.init(orig, bounding_box)

        optical_flow_output = None
        if self.run_optical_flow:
            optical_flow_output = self.of(orig)
            if optical_flow_output is not None:
                min_x, min_y, max_x, max_y = optical_flow_output
                min_x = int(min_x)
                min_y = int(min_y)
                max_x = int(max_x)
                max_y = int(max_y)
                flag = "normal"
                score = 1  # TODO THINK
            else:
                rospy.loginfo("failed to match features with optical flow")
        if (
            self.tracker_counter % self.tracker_run_iter == 0
            or optical_flow_output is None
        ):
            min_x, min_y, max_x, max_y, flag, score = self.tracker.run_frame(orig)
            w = max_x - min_x
            h = max_y - min_y
            self.of.roi = min_x, min_y, w, h
        bb_msg = Float32MultiArray()
        flag = 1 if flag == "normal" else 0
        bb_msg.data = [min_x, min_y, max_x, max_y, flag, score]
        self.bb_pub.publish(bb_msg)
        cv2.rectangle(modify_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 5)
        if self.clientsocket is not None:
            data = pickle.dumps(modify_image)
            message_size = struct.pack("L", len(data))  # unsigned long
            try:
                self.clientsocket.sendall(message_size + data)
                self.set_new_bounding_box(orig)
            except Exception as e:
                self.clientsocket = None

    def reconnect_to_socket(self):
        self.clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.clientsocket.connect((self.ip, self.port))
        except Exception as e:
            self.clientsocket = None

    def init_bounding_box(self, frame):
        data = pickle.dumps(frame)
        message_size = struct.pack("L", len(data))  # unsigned long
        try:
            self.clientsocket.sendall(message_size + data)
            ready_to_read, _, _ = select.select(
                [self.clientsocket], [], [], self.init_timeout
            )
            if ready_to_read:
                bounding_box = self.clientsocket.recv(1024)
                bounding_box = [
                    int(value) for value in bounding_box.decode().split(",")
                ]
                self.tracker.init_tracker(frame, bounding_box)
                self.init = True
                return bounding_box
        except Exception as e:
            self.clientsocket = None

    def set_new_bounding_box(self, frame):
        try:
            ready_to_read, _, _ = select.select([self.clientsocket], [], [], 0.01)
            if ready_to_read:
                temp = self.clientsocket.recv(1024)  # clear the buffer
                ready_to_read, _, _ = select.select(
                    [self.clientsocket], [], [], self.init_timeout
                )
                if ready_to_read:
                    bounding_box = self.clientsocket.recv(1024)
                    bounding_box = [
                        int(value) for value in bounding_box.decode().split(",")
                    ]
                    self.tracker.init_tracker(frame, bounding_box)
                    if self.run_optical_flow:
                        w = max_x - min_x
                        h = max_y - min_y
                        self.of.roi = min_x, min_y, w, h
        except Exception as e:
            self.clientsocket = None
