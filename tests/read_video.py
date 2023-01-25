import cv2
import sys

VIDEO_PATH = "video.mp4"
sys.path.append("/home/jetson/ros/pytracking/")

from object_tracker.object_tracker import ObjectTracker


class ReadImage:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = ObjectTracker()

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
            cv2.imshow(display_name, image)
            key = cv2.waitKey(1)
            # print(data)
            first = False


if __name__ == "__main__":
    ReadImage(VIDEO_PATH).run()
