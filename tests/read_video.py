import cv2
import sys



VIDEO_PATH = "bike_stand_fast.mp4"
#VIDEO_PATH =  "octagon_fly6.mp4"
#VIDEO_PATH = "out.mp4"
#VIDEO_PATH = "/tmp/video4.mp4"
# VIDEO_PATH="/media/jetson/6436-3639/beU/data/beu-data-stable/videos/carpet_fly7.mp4"
# VIDEO_PATH="/media/jetson/6436-3639/beU/data/beu-data-stable/videos/carpet_fly3.mp4"

sys.path.append("/home/jetson/ros/pytracking/")

from object_tracker.object_tracker import ObjectTracker


class ReadImage:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        # self.tracker = ObjectTracker(False, True, run_of_low_score=True)
        self.tracker = ObjectTracker(False, False)

    def run(self):
        display_name = "frame"
        first = True
        while True:
            ret, frame = self.cap.read()
            key = cv2.waitKey(1)
            if key == ord("r") or first:
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
            # print(data)
            first = False
            break


if __name__ == "__main__":
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    ReadImage(VIDEO_PATH).run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    file_name = VIDEO_PATH.split('.')[0]
    stats.dump_stats(file_name + '.prof')
