import pytest
import cv2
import sys
import numpy as np

PYTRACKING_PATH = "/home/jetson/ros/pytracking/"

VIDEO_PATH = PYTRACKING_PATH + "/tests/video.mp4"
sys.path.append(PYTRACKING_PATH)


from object_tracker.object_tracker import ObjectTracker


@pytest.mark.parametrize(
    "video_path, run_optical_flow, use_orb",
    [(VIDEO_PATH, False, False), (VIDEO_PATH, True, False), (VIDEO_PATH, True, True)],
)
def test_run_on_frame(video_path, run_optical_flow, use_orb, num_frame_run=20):
    cap = cv2.VideoCapture(video_path)
    tracker = ObjectTracker(run_optical_flow, use_orb)
    ret, frame = cap.read()
    x, y, w, h = [100, 100, 20, 20]
    tracker.init_bounding_box(frame, [x, y, w, h])
    for i in range(num_frame_run):
        ret, frame = cap.read()
        if i == 1 and use_orb == True:
            # set a random frame to enable orb detector
            frame = np.random.randint(255, size=frame.shape, dtype=np.uint8)
        image, data = tracker.run_frame(frame)