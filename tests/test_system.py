import pytest
import cv2
import numpy as np
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


PYTRACKING_PATH = str(Path(__file__).absolute().parent.parent)

sys.path.append(PYTRACKING_PATH)

EVALUATOR_PATH = PYTRACKING_PATH + '/beu-evaluator/'
CONFIG_PATH = PYTRACKING_PATH + '/pytracking_config.yaml'
TEST_DATA_VIDS = PYTRACKING_PATH + '/beu-data/videos/'
TEST_DATA_GT = PYTRACKING_PATH + '/beu-data/gt/'
TEST_RESULT_YAML = PYTRACKING_PATH + '/beu-data/test_results.yaml'
sys.path.append(EVALUATOR_PATH)
from beu_evaluator import evaluator
from beu_evaluator.algorithms.pytracking import ObjectTrackerAlgo
from beu_evaluator.utils import metrics, utils

def test_system():
    algo = ObjectTrackerAlgo(CONFIG_PATH)
    test_results_min = utils.read_yaml(TEST_RESULT_YAML)
    data = utils.match_gt_to_video(TEST_DATA_VIDS, TEST_DATA_GT)
    for video in data:
        gt = data[video]
        video_name = video.split('/')[-1].split('.')[0]
        logger.info(f"Testing video :{video_name}")
        ev = evaluator.Evaluator(algo, video, gt, run_mlflow=False)
        ev.add_metrics({"F1 threshold": metrics.metric_over_thresh(metrics.f1, 0.5)})
        results = ev.run()
        avg_f1 = utils.get_avg_list(results)['F1 threshold_avg']
        result_min = test_results_min[video_name]
        assert avg_f1 > result_min