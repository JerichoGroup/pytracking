import logging
import cv2
import collections
import numpy as np
from typing import List
from numpy import ndarray

Roi = collections.namedtuple("Roi", ["min_x", "min_y", "w", "h"])


class Matcher:
    class Params:
        def __init__(self):
            self.num_orb_features = 200
            self.bidirectional_enable = True
            self.bidirectional_thresh = 5.0
            self.min_points_for_find_homography = 40
            self.detection_params = dict(
                maxCorners=500, qualityLevel=0.01, minDistance=17, blockSize=7
            )
            self.tracking_params = dict(
                winSize=(15, 15),
                maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
                minEigThreshold=5e-4,
            )

    def __init__(self, use_orb: bool = False, params=Params()):
        """
        Args:
            use_orb: flag if use orb detector when optical flow falied.
            params: a Params class contains the parameters for the Matcher.
        """
        self._use_orb = use_orb
        self._params = params
        self._prev_image = None
        self._features = None
        self._orb = None
        self._bf = None
        if self._use_orb:
            self._orb = cv2.ORB_create()
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def init_bounding_box(self, image: ndarray, roi: List[int]):
        """
        this method sets a start roi for the tracker.
        Args:
            image: image
            roi: [x,y,w,h]
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.roi = Roi(min_x=roi[0], min_y=roi[1], w=roi[2], h=roi[3])
        self._set_new_features(image)
        logging.info("finish init matcher")

    def set_new_roi(self, roi):
        """
        this method sets a new roi
        Args:
            roi: [x,y,w,h]
        """
        self.roi = Roi(min_x=roi[0], min_y=roi[1], w=roi[2], h=roi[3])

    def _find_features(self, image: ndarray):
        """
        this method finds new features from the give frame
        """
        try:
            features = cv2.goodFeaturesToTrack(image, **self._params.detection_params)
            if features is None:
                logging.error("failed to match features using goodFeaturesToTrack")
                self._features = np.array([])
            else:
                self._features = np.squeeze(features)
        except Exception as e:
            logging.error(e)
            logging.error("goodFeaturesToTrack raise exception")
            self._features = np.array([])

    def _calc_orb(self, image: ndarray) -> ndarray:
        """
        this method calculates a homography matrix using BFMatcher.
        Args:
            image: image
        Returns:
            M(homography matrix).
            in case of failure return None.
        """
        kp1, des1 = self._orb.detectAndCompute(self._prev_image, None)
        kp2, des2 = self._orb.detectAndCompute(image, None)
        matches = self._bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[: self._params.num_orb_features]

        list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
        list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
        # if there are not enough features for homography return None
        if len(list_kp1) < self._params.min_points_for_find_homography:
            return None
        M, _ = cv2.findHomography(
            np.squeeze(list_kp1), np.squeeze(list_kp2), cv2.RANSAC, 5.0
        )
        return M

    def _calc_optical_flow(self, image: ndarray) -> ndarray:
        """
        this method calculates a homography matrix using OpticalFlowPyrLK.
        Args:
            image: image
        Returns:
            M(homography matrix)
            in case of failure return None.
        """
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
            self._prev_image,
            image,
            self._features,
            None,
            **self._params.tracking_params
        )
        if self._params.bidirectional_enable:
            p2, st2, err2 = cv2.calcOpticalFlowPyrLK(
                image, self._prev_image, p1, None, **self._params.tracking_params
            )
            proj_err = np.linalg.norm(p2 - self._features, axis=1)
            st = np.squeeze(st1 * st2) * (
                proj_err < self._params.bidirectional_thresh
            ).astype("uint8")
        else:
            st = np.squeeze(st1)
        # if there are not enough features for homography return None
        if (
            len(np.squeeze(self._features[st > 0]))
            < self._params.min_points_for_find_homography
        ):
            logging.error(
                "failed to find enough features with optical flow for homography"
            )
            return None
        M, _ = cv2.findHomography(
            np.squeeze(self._features[st > 0]), np.squeeze(p1[st > 0]), cv2.RANSAC, 5.0
        )
        return M

    def _calc_new_roi(self, M: ndarray) -> List[int]:
        """
        this method calculates a new roi using the M(homography matrix).
        Args:
             M(homography matrix)
        Returns:
            min_x, min_y, max_x, max_y
        """
        top_left = [self.roi.min_x, self.roi.min_y]
        top_right = [self.roi.min_x + self.roi.w, self.roi.min_y]
        bottom_right = [self.roi.min_x + self.roi.w, self.roi.min_y + self.roi.h]
        bottom_left = [self.roi.min_x, self.roi.min_y + self.roi.h]
        coord_of_roi = np.array(
            [top_left, top_right, bottom_right, bottom_left]
        ).reshape(4, 1, 2)
        transform_points = cv2.perspectiveTransform(
            coord_of_roi.astype(np.float), M
        ).reshape(4, 2)
        min_x = np.min(transform_points, axis=0)[0]
        min_y = np.min(transform_points, axis=0)[1]
        max_x = np.max(transform_points, axis=0)[0]
        max_y = np.max(transform_points, axis=0)[1]
        w = max_x - min_x
        h = max_y - min_y
        self.roi = Roi(min_x=min_x, min_y=min_y, w=w, h=h)
        return int(min_x), int(min_y), int(max_x), int(max_y)

    def _run_orb(self, image: ndarray) -> ndarray:
        """
        this method runs orb and returns homography matrix
        Args:
            image: image
        Returns:
            M(homography matrix)
            in case of failure return None.
        """
        M = None
        logging.info("trying orb")
        try:
            M = self._calc_orb(image)
            if M is None:
                logging.error("failed to match features with orb")
        except Exception as e:
            logging.error(e)
            logging.error("orb raise exception")
        return M

    def _run_optical_flow(self, image: ndarray) -> ndarray:
        """
        this method runs optical flow and returns homography matrix
        Args:
            image: image
        Returns:
            M(homography matrix)
            in case of failure return None.
        """
        M = None
        try:
            M = self._calc_optical_flow(image)
            if M is None:
                logging.error("failed to match features with optical flow")
        except Exception as e:
            logging.error(e)
            logging.error("optical flow raise exception")
        return M

    def _set_new_features(self, image: ndarray):
        """
        the method set a new features from the current frame
        and set the current image as the prev image.
        Args:
            image: image
        """
        self._find_features(image)
        self._prev_image = image

    def _run_orb_set_new_features(self, image: ndarray) -> ndarray:
        """
        this method runs orb (if the flag is set), after this the method
        set a new features from the current frame, and set the current image
        as the prev image.
        Args:
            image: image
        """
        M = None
        try:
            M = self._run_orb(image)
        except Exception as e:
            logging.error(e)
            logging.error("orb raise exception")
        self._set_new_features(image)
        return M

    def __call__(self, image: ndarray) -> List[int]:
        """
        this method runs optical flow and orb(if necessary) on the image.
        Args:
            image: image
        Returns:
            min_x, min_y, max_x, max_y
            in case of failure return None.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        M = None
        # check if there are enough features for using optical flow
        # if not and use_orb flag is set, the method tries to find
        # features using orb detector
        # if len(self._features) == 0 or len(self._features.shape) < 2:
        if True:
            if self._use_orb is True:
                M = self._run_orb_set_new_features(image)
                if M is None:
                    return None
                return self._calc_new_roi(M)
            else:
                self._set_new_features(image)
                return None
        M = self._run_optical_flow(image)
        if self._use_orb is True and M is None:
            M = self._run_orb_set_new_features(image)
        else:
            self._set_new_features(image)
        if M is None:
            return None
        return self._calc_new_roi(M)
