from collections import OrderedDict
from pytracking.evaluation import Tracker
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper



class CustomTracker(Tracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_frame(self, frame):
        """
        the method receive a frame as input and return
        [min_x, min_y, max_x, max_y], flag, score(float)
        args:
            frame: image
        """
        if frame is None:
            return None
        out = self.tracker.track(frame)
        state = [int(s) for s in out["target_bbox"][1]]
        min_x = state[0]
        min_y = state[1]
        max_x = state[2] + state[0]
        max_y = state[3] + state[1]
        return min_x, min_y, max_x, max_y, out["flag"][1], out["score"][1]

    def init_tracker(self, frame, bounding_box):
        """
        the method init the tracker
        args:
            frame: image
            bounding_box: [x,y,w,h]
        """
        params = self.get_parameters()
        debug_ = getattr(params, "debug", 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        multiobj_mode = getattr(
            params,
            "multiobj_mode",
            getattr(self.tracker_class, "multiobj_mode", "default"),
        )
        if multiobj_mode == 'default':
            self.tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                self.tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            self.tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        def _build_init_info(box):
            return {
                "init_bbox": OrderedDict({1: box}),
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
