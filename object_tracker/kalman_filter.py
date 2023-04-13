import cv2
import numpy as np

class KalmanFilter:
    class Params:
        def __init__(self):
            f = 0.01
            self.drag_factor = 1
            self.initial_estimate_error = [1*f, 1*f, 1*f]
            self.motion_noise = [10000*f, 10000*f]
            self.measurement_noise = 1000*f
            self.measurement_noise1 = 10000*f
            self.box_noise = 250*f
            self.box_noise1 = 250*f

    def __init__(self, state = np.array([0,0,0,0,0,0]),params = Params()):
        '''
        reminder:
        X - state
        U - control
        w - process noise
        Z - measurements
        v - measurement noise
        model:
        X_k+1 =  transitionMatrix*X_t + controlMatrix*U_t + w_k
        Z_k = measurementMatrix*X_k + v_k

        in our case:
        X = [x,y,vx,vy,w,h]
        x_k+1 = x_k + vx_k
        vx_k+1 = drag_factor*vx_k
        w_k+1 = w_k
        U = None
        Z = [x_det, y_det, w_det, h_det]
        x_k = x_det
        '''

        self._params = params
        kalman = cv2.KalmanFilter(6, 4)
        kalman.statePost = state.astype(np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1]], np.float32)
        d = self._params.drag_factor
        kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0],
                                            [0, 1, 0, 1, 0, 0],
                                            [0, 0, d, 0, 0, 0],
                                            [0, 0, 0, d, 0, 0],
                                            [0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 1]], np.float32)

        kalman.controlMatrix = None

        kalman.processNoiseCov = np.array([[self._params.motion_noise[0], 0, 0, 0, 0, 0],
                                           [0, self._params.motion_noise[0], 0, 0, 0, 0],
                                           [0, 0, self._params.motion_noise[1], 0, 0, 0],
                                           [0, 0, 0, self._params.motion_noise[1], 0, 0],
                                           [0,0,0,0,self._params.box_noise,0],
                                           [0,0,0,0,0,self._params.box_noise]], np.float32)
        kalman.measurementNoiseCov = np.array([[self._params.measurement_noise, 0, 0, 0],
                                               [0, self._params.measurement_noise, 0, 0],
                                               [0,0,self._params.box_noise,0],
                                               [0,0,0,self._params.box_noise]], np.float32)
        kalman.errorCovPre = np.array([[self._params.initial_estimate_error[0], 0, 0, 0, 0, 0],
                                           [0, self._params.initial_estimate_error[0], 0, 0, 0, 0],
                                           [0, 0, self._params.initial_estimate_error[1], 0, 0, 0],
                                           [0, 0, 0, self._params.initial_estimate_error[1], 0, 0],
                                           [0, 0, 0, 0, self._params.initial_estimate_error[2], 0],
                                           [0, 0, 0, 0, 0, self._params.initial_estimate_error[2]]], np.float32)
        self._kf = kalman

    def correct(self, measurement):
        self._kf.measurementNoiseCov = np.array([[self._params.measurement_noise, 0, 0, 0],
                                               [0, self._params.measurement_noise, 0, 0],
                                               [0,0,self._params.box_noise,0],
                                               [0,0,0,self._params.box_noise]], np.float32)
        return self._kf.correct(measurement.astype(np.float32))[[0,1,4,5]].astype(float).flatten()

    def correct1(self, measurement):
        self._kf.measurementNoiseCov = np.array([[self._params.measurement_noise1, 0, 0, 0],
                                               [0, self._params.measurement_noise1, 0, 0],
                                               [0,0,self._params.box_noise1,0],
                                               [0,0,0,self._params.box_noise1]], np.float32)
        return self._kf.correct(measurement.astype(np.float32))[[0,1,4,5]].astype(float).flatten()

    def predict(self):
        return self._kf.predict()[[0,1,4,5]].astype(float).flatten()
