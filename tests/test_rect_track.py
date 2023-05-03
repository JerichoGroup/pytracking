
import cv2
import numpy as np


import read_video

TMP_VID_PATH = '/tmp/test_vid.mp4'

def create_vid(vid_length = 100, image_size:int = 400, top_left_x_start = 170, top_left_y_start = 170, w = 20, h =20, step_size = 10):

    image = np.zeros((image_size, image_size), dtype=np.uint8)
    image[ top_left_y_start : top_left_y_start + h ,top_left_x_start : top_left_x_start + w ] = 255

    images = [image]
    gt_loc = [[top_left_y_start, top_left_x_start]]
    vid = cv2.VideoWriter(TMP_VID_PATH, cv2.VideoWriter_fourcc(*'DIVX'), 9, (image_size, image_size), False)
    vid.write(image)


    for i in range(vid_length):
        step_x = 0
        step_y = 0
        rand_move = np.random.uniform(0, 1)
        pos_neg_move = np.random.uniform(0, 1)
        
        if pos_neg_move <0.5:
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
        top_y = gt_loc[-1][0] + step_y * direction_step
        top_x = gt_loc[-1][1] + step_x * direction_step
        gt_loc.append([top_y, top_x])
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        image[ top_y: top_y + h ,top_x : top_x + w] = 255
        # images.append(image)
        vid.write(image)

    vid.release()

    return gt_loc




if __name__ == "__main__":
    gt_loc = create_vid()
    read_video.ReadImage(TMP_VID_PATH).run()

    


