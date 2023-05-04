import logging

from object_tracker.custom_tracker import CustomTracker
from object_tracker.match import Matcher
from typing import List, Tuple
from numpy import ndarray
from omegaconf import DictConfig
logging.getLogger().setLevel(logging.INFO)


class ObjectTracker:

    MAX_COUNTER = 10

    def __init__(
        self,
        cfg: DictConfig
        # run_optical_flow: bool = False,
        # use_orb: bool = False,
        # tracker_run_iter: int = 3,
        # run_of_low_score: bool = False,
        # score_thresh: float = 0.4,
        # name: str = "dimp",
        # param_name: str = "dimp18"

        ):
        """
        Args:
            run_optical_flow: a flag if to run optical flow(using the matcher class)
            use_orb: a flag if to use orb when optical flow falied
            tracker_run_iter: the number of times to run matcher between deep-tracker run.
        """
        self._run_optical_flow = cfg.run_optical_flow
        self.run_of_low_score = cfg.run_of_low_score
        self.score_thresh = cfg.score_thresh
        self._tracker_run_iter = max(cfg.tracker_run_iter, 1)
        self._tracker_counter = 0
        self._tracker = CustomTracker(cfg.name, cfg.param_name)
        self._match = None
        self._init = False
        if self._run_optical_flow or self.run_of_low_score:
            self._match = Matcher(cfg.Matcher, cfg.use_orb)
    

    def is_tracker_ready(self) -> bool:
        """
        this method return a True if the tracker is ready( the tracker ready after
        init_bounding_box method is called).
        """
        return self._init

    def run_frame(self, img: ndarray) -> Tuple[ndarray, List[int], str, float]:
        """
        this method return a bounding flag and score on the given frame.
        return value: [x,y,w,h], flag(1 or 0), score(float)
        in case of falure the method returns the image, None
        Args:
            img: image
        Returns:
            img, [x,y,w,h], flag(1 or 0), score
        """
        if not self.is_tracker_ready():
            return img, None
        if self._tracker_counter > ObjectTracker.MAX_COUNTER:
            self._tracker_counter = 0
        self._tracker_counter += 1
        orig = img.copy()
        optical_flow_output = None
        if self._run_optical_flow:
            if self._tracker_counter % self._tracker_run_iter != 0:
                try:
                    optical_flow_output = self._match(orig)
                except Exception as e:
                    logging.error(e)
                    logging.error("match raise exception")
                if optical_flow_output is not None:
                    min_x, min_y, max_x, max_y = optical_flow_output
                    flag = "normal"
                    score = 0
                else:
                    logging.error("failed to match features with matcher")
        if (
            self._tracker_counter % self._tracker_run_iter == 0
            or optical_flow_output is None
        ):
            tracker_output = self._tracker.run_frame(orig)
            if tracker_output is None:
                return img, None
            
            min_x, min_y, max_x, max_y, flag, score = tracker_output



            if self.run_of_low_score:
                if self._tracker_counter % self._tracker_run_iter != 0:
                    try:
                        optical_flow_output = self._match(orig)
                    except Exception as e:
                        logging.error(e)
                        logging.error("match raise exception")
                    if optical_flow_output is not None:
                        if score < self.score_thresh:
                            logging.info(f'Score is low, running OF {score}')
                            min_x, min_y, max_x, max_y = optical_flow_output
                            flag = "normal"
                            score = 0
                    else:
                        logging.error("failed to match features with matcher")
                
                
                

            w = max_x - min_x
            h = max_y - min_y
            if self._run_optical_flow or self.run_of_low_score:
                self._match.set_new_roi([min_x, min_y, w, h])

        flag = 1 if flag == "normal" else 0
        # x, y, w, h, was_frame_situation_algo_wise_was_normal, score
        data = [min_x, min_y, max_x - min_x, max_y - min_y, flag, score]
        return img, data

    def init_bounding_box(self, frame: ndarray, bounding_box: List[int]):
        """
        this method init the first bounding box for the tracker.
        Args:
            frame: image
            bounding_box: [x,y,w,h]
        """
        self._tracker.init_tracker(frame, bounding_box)
        logging.info("finish init tracker bounding box")
        logging.info(f"bounding box: {bounding_box}")
        self._init = True
        if self._run_optical_flow or self.run_of_low_score:
            self._match.init_bounding_box(frame, bounding_box)

    def get_parameters(self):
        return self._tracker.get_parameters()
