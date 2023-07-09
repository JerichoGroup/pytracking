#Simple way to run or profile the run time of a video with the tracker.
import cv2
import sys
from pathlib import Path
import cProfile
import pstats

PYTRACKING_PATH = str(Path(__file__).absolute().parent.parent)

sys.path.append(PYTRACKING_PATH)
CONFIG_PATH = PYTRACKING_PATH + "/pytracking_config.yaml"

VIDEO_PATH = "bike_stand_fast.mp4"
#VIDEO_PATH =  "octagon_fly6.mp4"
#VIDEO_PATH = "out.mp4"
#VIDEO_PATH = "/tmp/video4.mp4"
# VIDEO_PATH="/media/jetson/6436-3639/beU/data/beu-data-stable/videos/carpet_fly7.mp4"
# VIDEO_PATH="/media/jetson/6436-3639/beU/data/beu-data-stable/videos/carpet_fly3.mp4"

from utils import utils
from object_tracker.object_tracker import ObjectTracker


class ReadImage:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        cfg = utils.read_yaml(CONFIG_PATH)
        self.tracker = ObjectTracker(cfg.ObjectTracker)
    

    def run(self, gt_lov = None):
        display_name = "frame"
        while True:
            ret, frame = self.cap.read()
            key = cv2.waitKey(1)
            if key == ord("r"):
                cv2.putText(
                    frame,
                    "Select target ROI and press ENTER",
                    (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (0, 0, 0),
                    1,
                )
                cv2.imshow(display_name, frame)
                x, y, w, h = cv2.selectROI(display_name, frame, fromCenter=False)
                self.tracker.init_bounding_box(frame, [x, y, w, h])

            if not ret:
                break
            image, data = self.tracker.run_frame(frame)
            min_x , min_y , w , h, _, _ = data
            image = cv2.rectangle(image, (min_x, min_y), (min_x + w, min_y + h), (0, 255, 0), 5)
            cv2.imshow(display_name, image)
            key = cv2.waitKey(1)
            


if __name__ == "__main__":
    # In order to run profiling uncomment the code.


    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    reader = ReadImage(VIDEO_PATH)
    reader.run()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # file_name = VIDEO_PATH.split('.')[0]
    # stats.dump_stats(file_name + '.prof')
