import pytest
import cv2
import sys
import numpy as np
from mock import patch
from pathlib import Path

PYTRACKING_PATH = str(Path(__file__).absolute().parent.parent)

sys.path.append(PYTRACKING_PATH)

VIDEO_PATH = PYTRACKING_PATH + "/tests/video.mp4"
CONFIG_PATH = PYTRACKING_PATH + "/pytracking_config.yaml"
from utils import utils

from object_tracker.object_tracker import ObjectTracker
from object_tracker.match import Matcher


@pytest.mark.parametrize(
    "video_path, run_optical_flow, use_orb",
    [(VIDEO_PATH, True, True)],
)
def test_raise_exception_in_matcher(
    video_path, run_optical_flow, use_orb, num_frame_run=3
):
    """
    this test run a video on the tracker
    """
    with patch(
        "object_tracker.match.Matcher.__call__",
        side_effect=Exception("optical flow falied"),
    ) as mocked_object:
        cap = cv2.VideoCapture(video_path)
        cfg = utils.read_yaml(CONFIG_PATH)
        tracker = ObjectTracker(cfg.ObjectTracker)
        ret, frame = cap.read()
        x, y, w, h = [100, 100, 20, 20]
        tracker.init_bounding_box(frame, [x, y, w, h])
        for i in range(num_frame_run):
            ret, frame = cap.read()
            image, data = tracker.run_frame(frame)

@pytest.mark.parametrize(
    "video_path, run_optical_flow, use_orb",
    [(VIDEO_PATH, False, False), (VIDEO_PATH, True, False), (VIDEO_PATH, True, True)],
)
def test_run_on_frame(video_path, run_optical_flow, use_orb, num_frame_run=20):
    """
    this test run a video on the tracker
    """
    cap = cv2.VideoCapture(video_path)
    cfg = utils.read_yaml(CONFIG_PATH)
    tracker = ObjectTracker(cfg.ObjectTracker)
    ret, frame = cap.read()
    x, y, w, h = [100, 100, 20, 20]
    tracker.init_bounding_box(frame, [x, y, w, h])
    for i in range(num_frame_run):
        ret, frame = cap.read()
        image, data = tracker.run_frame(frame)


@pytest.mark.parametrize(
    "video_path, run_optical_flow, use_orb",
    [(VIDEO_PATH, True, True)],
)
def test_run_on_frame_using_orb(
    video_path, run_optical_flow, use_orb, num_frame_run=20
):
    """
    this test make the optical flow to fail to trigger the use of orb detector
    """
    cap = cv2.VideoCapture(video_path)
    cfg = utils.read_yaml(CONFIG_PATH)
    tracker = ObjectTracker(cfg.ObjectTracker)
    ret, frame = cap.read()
    x, y, w, h = [100, 100, 20, 20]
    tracker.init_bounding_box(frame, [x, y, w, h])
    for i in range(num_frame_run):
        ret, frame = cap.read()
        if i == 1 and use_orb == True:
            # set a random frame to enable orb detector
            frame = np.random.randint(255, size=frame.shape, dtype=np.uint8)
        image, data = tracker.run_frame(frame)


@pytest.mark.parametrize(
    "video_path, run_optical_flow, use_orb",
    [(VIDEO_PATH, False, False), (VIDEO_PATH, True, False), (VIDEO_PATH, True, True)],
)
def test_run_on_random_frame(video_path, run_optical_flow, use_orb, num_frame_run=20):
    """
    this test run a video on the tracker
    """
    cap = cv2.VideoCapture(video_path)
    cfg = utils.read_yaml(CONFIG_PATH)
    tracker = ObjectTracker(cfg.ObjectTracker)
    ret, frame = cap.read()
    x, y, w, h = [100, 100, 20, 20]
    tracker.init_bounding_box(frame, [x, y, w, h])
    for i in range(num_frame_run):
        frame = np.random.randint(255, size=frame.shape, dtype=np.uint8)
        image, data = tracker.run_frame(frame)
