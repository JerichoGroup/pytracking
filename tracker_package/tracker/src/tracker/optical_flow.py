import cv2
import numpy as np

class VisualTrackerKLT:
    class Params:
        def __init__(self):

            self.bidirectional_enable = True
            self.bidirectional_thresh = 2.0
            self.min_points_for_find_homography = 10
            self.detection_params = dict(
                maxCorners=500, qualityLevel=0.01, minDistance=21, blockSize=7
            )
            self.tracking_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                minEigThreshold=5e-4,
            )

    def __init__(self, params=Params()):
        self._params = params
        self._prev_pyr = None
        self._kernel_d = np.ones(
            (
                self._params.detection_params["minDistance"],
                self._params.detection_params["minDistance"],
            ),
            dtype=np.uint8,
        )
        self._frame_num = 0

    def init(self, image, roi):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._prev_pyr = gray
        self.roi = roi
        self.features = {}
        #self.features = cv2.goodFeaturesToTrack(gray, **self._params.detection_params)

    def __call__(self, features):
        current_features = {}
        for feat in features.features:
            current_features[feat.id] = [feat.x, feat.y]

        keys = current_features.keys() & self.features.keys()
        p1 = []
        p2 = []
        for key in keys:
            p1.append(self.features[key])
            p2.append(current_features[key])
        print(len(keys))
        if len(keys) < 4:
            self.features = current_features
            return None
        p1 = np.array(p1)
        p2 = np.array(p2)
        M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if M is None:
            self.features = current_features
            return None
        points = []
        point1 = [self.roi[0], self.roi[1]]
        point2 = [self.roi[0] + self.roi[2], self.roi[1]]
        point3 = [self.roi[0] + self.roi[2], self.roi[1] + self.roi[3]]
        point4 = [self.roi[0], self.roi[1] + self.roi[3]]
        coord_of_roi = np.array([point1, point2, point3, point4]).reshape(-1, 1, 2)
        transform_points = cv2.perspectiveTransform(
            coord_of_roi.astype(np.float), M
        ).reshape(-1, 2)
        min_x = np.min(transform_points, axis=0)[0]
        min_y = np.min(transform_points, axis=0)[1]
        max_x = np.max(transform_points, axis=0)[0]
        max_y = np.max(transform_points, axis=0)[1]
        w = max_x - min_x
        h = max_y - min_y
        self.roi = [min_x, min_y, w, h]
        self.features = current_features
        return min_x, min_y, max_x, max_y
