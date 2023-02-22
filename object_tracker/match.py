import logging
import cv2
import collections
import numpy as np

Roi = collections.namedtuple("Roi", ["min_x", "min_y", "w", "h"])


class Matcher:
    class Params:
        def __init__(self):
            self.num_orb_features = 100
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

    def __init__(self, use_orb=False, params=Params()):
        self.orb = cv2.ORB_create()
        self.use_orb = use_orb
        self._params = params
        self._prev_image = None
        self.features = None

    def init(self, image, roi):
        """
        this method sets a start roi for the tracker
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._prev_image = image
        self.roi = Roi(min_x=roi[0], min_y=roi[1], w=roi[2], h=roi[3])
        self._find_features(image)
        logging.info("finish init matcher")

    def set_new_roi(self, roi):
        self.roi = Roi(min_x=roi[0], min_y=roi[1], w=roi[2], h=roi[3])

    def _find_features(self, image):
        """
        this method finds new features from the give frame
        """
        features = cv2.goodFeaturesToTrack(image, **self._params.detection_params)
        if features is None:
            logging.error("failed to match features using goodFeaturesToTrack")
            self.features = []
        else:
            self.features = features

    def _calc_orb(self, image):
        """
        this method calculates a homography matrix using BFMatcher
        return M(homography matrix), mask
        """
        kp1, des1 = self.orb.detectAndCompute(self._prev_image, None)
        kp2, des2 = self.orb.detectAndCompute(image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[: self._params.num_orb_features]

        list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
        list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
        if len(list_kp1) < self._params.min_points_for_find_homography:
            return None, None
        M, mask = cv2.findHomography(
            np.squeeze(list_kp1), np.squeeze(list_kp2), cv2.RANSAC, 5.0
        )
        return M, mask

    def _calc_optical_flow(self, p0, image):
        """
        this method calculates a homography matrix using OpticalFlowPyrLK
        return M(homography matrix), mask
        """
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
            self._prev_image, image, p0, None, **self._params.tracking_params
        )
        if self._params.bidirectional_enable:
            p2, st2, err2 = cv2.calcOpticalFlowPyrLK(
                image, self._prev_image, p1, None, **self._params.tracking_params
            )
            proj_err = np.linalg.norm(p2 - p0, axis=1)
            st = np.squeeze(st1 * st2) * (
                proj_err < self._params.bidirectional_thresh
            ).astype("uint8")
        else:
            st = np.squeeze(st1)
        if len(np.squeeze(p0[st > 0])) < self._params.min_points_for_find_homography:
            logging.error("failed to match features with optical flow")
            return None, None
        M, mask = cv2.findHomography(
            np.squeeze(p0[st > 0]), np.squeeze(p1[st > 0]), cv2.RANSAC, 5.0
        )
        return M, mask

    def _calc_new_roi(self, M):
        """
        this method calculates a new roi using the M(homography matrix).
        return: min_x, min_y, max_x, max_y
        """
        top_left = [self.roi.min_x, self.roi.min_y]
        top_right = [self.roi.min_x + self.roi.w, self.roi.min_y]
        bottom_right = [self.roi.min_x + self.roi.w, self.roi.min_y + self.roi.h]
        bottom_left = [self.roi.min_x, self.roi.min_y + self.roi.h]
        coord_of_roi = np.array([top_left, top_right, bottom_right, bottom_left]).reshape(-1, 1, 2)
        transform_points = cv2.perspectiveTransform(
            coord_of_roi.astype(np.float), M
        ).reshape(-1, 2)
        min_x = np.min(transform_points, axis=0)[0]
        min_y = np.min(transform_points, axis=0)[1]
        max_x = np.max(transform_points, axis=0)[0]
        max_y = np.max(transform_points, axis=0)[1]
        w = max_x - min_x
        h = max_y - min_y
        self.roi = Roi(min_x=min_x, min_y=min_y, w=w, h=h)
        return min_x, min_y, max_x, max_y

    def __call__(self, image):
        """
        this method runs optical flow and orb(if necessary) on the image
        return: min_x, min_y, max_x, max_y
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p0 = np.squeeze(self.features)

        if len(p0) == 0 or len(p0.shape) < 2:
            self._find_features(image)
            self._prev_image = image
            return None
        try:
            M, mask = self._calc_optical_flow(p0, image)
        except Exception as e:
            logging.error(e)
            logging.error("optical flow falied")
            self._find_features(image)
            self._prev_image = image
            return None
        self._find_features(image)
        self._prev_image = image
        if M is None:
            if self.use_orb is True:
                logging.info("trying orb")
                M, mask = self._calc_orb(image)
                if M is None:
                    logging.error("failed to match features with orb")
                    return None
            else:
                return None
        return self._calc_new_roi(M)
