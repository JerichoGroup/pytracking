import pytest
import cv2
import numpy as np
import sys
from pathlib import Path

PYTRACKING_PATH = str(Path(__file__).absolute().parent.parent)

sys.path.append(PYTRACKING_PATH)

EVALUATOR_PATH = PYTRACKING_PATH + '/beu-evaluator/'
CONFIG_PATH = PYTRACKING_PATH + '/pytracking_config.yaml'
sys.path.append(EVALUATOR_PATH)
from beu_evaluator import evaluator
from beu_evaluator.algorithms.pytracking import ObjectTrackerAlgo
from beu_evaluator.utils import metrics, utils

import read_video


TMP_VID_PATH = '/tmp/test_vid.mp4'
TMP_GT_PATH = '/tmp/test_gt.gt'

def rotate_rect(points, angle_rad):
    angle = np.deg2rad(angle_rad)
    c_x, c_y = np.mean(points, axis=0)
    rot_point = []
    for px, py in points:       
       rot_point.append([c_x + np.cos(angle) * (px - c_x) - np.sin(angle) * (py - c_x),
                         c_y + np.sin(angle) * (px - c_y) + np.cos(angle) * (py - c_y)])
    return np.array(rot_point, dtype=np.int)

def create_vid(vid_length = 100, image_size:int = 600, top_left_x_start = 270, top_left_y_start = 270, w = 20, h =20,
               step_size = 5, angle_rang = [-20, 20]):

    image = np.zeros((image_size, image_size), dtype=np.uint8)
    bot_right_y = top_left_y_start + h
    bot_right_x = top_left_x_start + w
    image[top_left_y_start : bot_right_y, top_left_x_start : bot_right_x] = 255

    images = [image]
    gt_loc = [[top_left_x_start, top_left_y_start, h, w ]]
    vid = cv2.VideoWriter(TMP_VID_PATH, cv2.VideoWriter_fourcc(*'DIVX'), 9, (image_size, image_size), False)
    vid.write(image)
    for i in range(vid_length):
        step_x = 0
        step_y = 0
        rand_move = np.random.uniform(0, 1)
        pos_neg_move = np.random.uniform(0, 1)
        
        if pos_neg_move <= 0.5:
            direction_step = 1
        else:
            direction_step = -1

        if rand_move < 0.3:
            step_x = step_size    
        if rand_move < 0.6 and rand_move > 0.3:
            step_y = step_size
        if rand_move < 0.9 and rand_move > 0.6:
            step_x = step_size
            step_y = step_size

        top_y = gt_loc[-1][1] + step_y * direction_step
        top_x = gt_loc[-1][0] + step_x * direction_step
        bot_y = top_y + h
        bot_x = top_x + w
        cords = np.array([[top_x, top_y], [top_x, bot_y], [bot_x, bot_y], [bot_x, top_y]])

        cords_rotated = rotate_rect(cords, np.random.uniform(angle_rang[0], angle_rang[1]))
        gt_h = np.max(cords_rotated, axis=0)[0] - np.min(cords_rotated, axis=0)[0]
        gt_w = np.max(cords_rotated, axis=0)[1] - np.min(cords_rotated, axis=0)[1]
        gt_loc.append([np.min(cords_rotated, axis=0)[0], np.min(cords_rotated, axis=0)[1],
                       gt_h, gt_w])
        
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.fillPoly(image, pts=[cords_rotated], color=255)
        # image = cv2.rectangle(image, [gt_loc[-1][0] ,gt_loc[-1][1]], [gt_loc[-1][2] ,gt_loc[-1][3]], (0, 120, 0), 5)
        images.append(image)
        vid.write(image)

    vid.release()
    np.savetxt(TMP_GT_PATH, gt_loc)

    

def test_rect():
    create_vid()
    algo = ObjectTrackerAlgo(CONFIG_PATH)
    ev = evaluator.Evaluator(alg=algo, video_path=TMP_VID_PATH, gt_path=TMP_GT_PATH, run_mlflow=False)
    ev.add_metrics({"IOU threshold": metrics.metric_over_thresh(metrics.iou, 0.8)})
    results = ev.run()
    avg_iou = utils.get_avg_list(results)['IOU threshold_avg']
    
    assert avg_iou > 0.7


if __name__ == '__main__':
    test_rect()

    


