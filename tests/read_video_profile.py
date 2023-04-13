import cv2
import sys
from vprof import runner

VIDEO_PATH = "bike_stand_fast.mp4"
sys.path.append("/home/jetson/ros/pytracking/")

from object_tracker.object_tracker import ObjectTracker


class ReadImage:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = ObjectTracker()

    def run(self):
        display_name = "frame"
        first = True
        counter = 0 
        while True:
            counter += 1
            ret, frame = self.cap.read()
            if first:
                x, y, w, h = 358, 296, 35, 44
                self.tracker.init_bounding_box(frame, [x, y, w, h])
            if not ret:
                break
            image, data = self.tracker.run_frame(frame)
            first = False
            if counter == 10:
                return


if __name__ == "__main__":
    obj = ReadImage(VIDEO_PATH)
    runner.run(obj.run, 'cmhp', host='localhost', port=8005)
