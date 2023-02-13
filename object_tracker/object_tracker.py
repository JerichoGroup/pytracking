import logging
from pytracking.evaluation import Tracker
from object_tracker.match import Matcher


class ObjectTracker:

    MAX_COUNTER = 10

    def __init__(self, run_optical_flow=False, use_orb=False, tracker_run_iter=3):
        self._run_optical_flow = run_optical_flow
        self._tracker_run_iter = max(tracker_run_iter, 1)
        self._tracker_counter = 0
        self._tracker = Tracker("dimp", "dimp18")
        self._match = Matcher(use_orb)
        self.init = False

    def run_frame(self, img):
        if not self.init:
            return
        if self._tracker_counter > ObjectTracker.MAX_COUNTER:
            self._tracker_counter = 0
        self._tracker_counter += 1
        orig = img.copy()
        optical_flow_output = None
        if self._run_optical_flow:
            optical_flow_output = self._match(orig)
            if optical_flow_output is not None:
                min_x, min_y, max_x, max_y = optical_flow_output
                min_x = int(min_x)
                min_y = int(min_y)
                max_x = int(max_x)
                max_y = int(max_y)
                flag = "normal"
                score = 1
            else:
                logging.error("failed to match features")
        if (
            self._tracker_counter % self._tracker_run_iter == 0
            or optical_flow_output is None
        ):
            min_x, min_y, max_x, max_y, flag, score = self._tracker.run_frame(orig)
            w = max_x - min_x
            h = max_y - min_y
            self._match.set_new_roi([min_x, min_y, w, h])

        flag = 1 if flag == "normal" else 0
        # x, y, w, h, was_frame_situation_algo_wise_was_normal, score
        data = [min_x, min_y, max_x - min_x, max_y - min_y, flag, score]
        return img, data

    def init_bounding_box(self, frame, bounding_box):
        self._tracker.init_tracker(frame, bounding_box)
        logging.info("finish init tracker bounding box")
        logging.info(f"bounding box: {bounding_box}")
        self.init = True
        if self._run_optical_flow:
            self._match.init(frame, bounding_box)
