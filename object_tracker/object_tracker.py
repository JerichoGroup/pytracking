import cv2
import numpy as np
import sys
import struct
import torch
import time

from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation import Tracker
from object_tracker.match import Matcher


class ObjectTracker:

    MAX_COUNTER = 10

    def __init__(self, run_optical_flow=True, tracker_run_iter=3):
        self.run_optical_flow = run_optical_flow
        self.tracker_run_iter = tracker_run_iter
        self.tracker_counter = 0
        self.tracker = Tracker("dimp", "dimp18")
        self.match = Matcher()
        self.init = False

    def run_frame(self, img):
        if not self.init:
            return
        if self.tracker_counter > ObjectTracker.MAX_COUNTER:
            self.tracker_counter = 0
        self.tracker_counter += 1
        orig = img.copy()
        optical_flow_output = None
        if self.run_optical_flow:
            optical_flow_output = self.match(orig)
            if optical_flow_output is not None:
                min_x, min_y, max_x, max_y = optical_flow_output
                min_x = int(min_x)
                min_y = int(min_y)
                max_x = int(max_x)
                max_y = int(max_y)
                flag = "normal"
                score = 1
            else:
                print("failed to match features")
        if (
            self.tracker_counter % self.tracker_run_iter == 0
            or optical_flow_output is None
        ):
            # start_time = time.time()
            min_x, min_y, max_x, max_y, flag, score = self.tracker.run_frame(orig)
            # print("netowrk")
            # print(time.time() - start_time)
            w = max_x - min_x
            h = max_y - min_y
            self.match.roi = min_x, min_y, w, h
        flag = 1 if flag == "normal" else 0
        data = [min_x, min_y, max_x, max_y, flag, score]
        #cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 5)
        return img, data

    def init_bounding_box(self, frame, bounding_box):
        self.tracker.init_tracker(frame, bounding_box)
        self.init = True
        if self.run_optical_flow:
            self.match.init(frame, bounding_box)
