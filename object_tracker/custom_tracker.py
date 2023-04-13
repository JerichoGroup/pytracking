from collections import OrderedDict
from pytracking.evaluation import Tracker
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from typing import List, Tuple
from numpy import ndarray


class CustomTracker(Tracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_frame(self, frame: ndarray) -> Tuple[List[int], str, float]:
        """
        the method receive a frame as input and run the tracker on it.
        Args:
            frame: image
        Returns:
            [min_x, min_y, max_x, max_y], flag, score
        """
        if frame is None:
            return None
        out = self.tracker.track(frame)
        state = [int(s) for s in out["target_bbox"]]
        min_x = state[0]
        min_y = state[1]
        max_x = state[2] + state[0]
        max_y = state[3] + state[1]

        return min_x, min_y, max_x, max_y, out["flag"], out["score"]

    def init_tracker(self, frame, bounding_box: List[int]):
        """
        the method init the tracker
        Args:
            frame: image
            bounding_box: [x,y,w,h]
        """
        params = self.get_parameters()
        debug_ = getattr(params, "debug", 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        # dimp can work also in "multiobj_mode", but in out usage we use default mode
        multiobj_mode = "default"
        self.tracker = self.create_tracker(params)
        if hasattr(self.tracker, "initialize_features"):
            self.tracker.initialize_features()

        def _build_init_info(box):
            return {
                "init_bbox": box,
                "init_object_ids": [
                    1,
                ],
                "object_ids": [
                    1,
                ],
                "sequence_object_ids": [
                    1,
                ],
            }

        if bounding_box is not None:
            assert isinstance(bounding_box, (list, tuple))
            assert len(bounding_box) == 4, "valid box's foramt is [x,y,w,h]"
        self.tracker.initialize(frame, _build_init_info(bounding_box))
