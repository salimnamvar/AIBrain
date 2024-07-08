"""Pose Tracker Module

"""

# region Imported Dependencies
from typing import List

import numpy as np
from filterpy.kalman import KalmanFilter

from brain.utils.cv.shape.ps import Pose2D
from brain.utils.ml.seg import SegBBox2D
from brain.utils.obj import BaseObject, BaseObjectList
# endregion Imported Dependencies


class State(BaseObject):
    def __init__(self, a_box: SegBBox2D, a_pose: Pose2D):
        super().__init__()
        self.box = a_box
        self.pose = a_pose

    def to_dict(self) -> dict:
        dic = {"box": self.box, "pose": self.pose}
        return dic


class StateList(BaseObjectList[State]):
    def __init__(
        self,
        a_name: str = "StateList",
        a_max_size: int = -1,
        a_items: List[State] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


class Tracker:
    def __init__(self, a_state: State, a_conf_thre: float, a_num_kps: int = 17):
        self.num_kps = a_num_kps
        self.conf_thre: float = a_conf_thre
        self.dim_state: int = 4  # Dimension of the state (x, y, vx, vy)
        self.dim_measure: int = 2  # Dimension of the measurement (x, y)
        self.states: StateList = StateList()
        self.update_age: int = 0
        self.states.append(a_state)
        self.kf = KalmanFilter(
            dim_x=self.num_kps * self.dim_state,
            dim_z=self.num_kps * self.dim_measure,
        )
        self._init_kf(a_state=a_state)

    def _init_kf(self, a_state: State):
        # Initialize the state transition matrix
        F = np.eye(self.num_kps * self.dim_state)
        for i in range(self.num_kps):
            idx = 4 * i
            F[idx : idx + 2, idx + 2 : idx + 4] = np.eye(2)  # position to velocity
        self.kf.F = F

        # Define the measurement matrix
        H = np.zeros((2 * self.num_kps, self.num_kps * self.dim_state))
        for i in range(self.num_kps):
            idx = 4 * i
            H[2 * i : 2 * i + 2, idx : idx + 2] = np.eye(2)  # x, y positions
        self.kf.H = H

        # Set up the process noise covariance matrix
        transition_covariance_noise = 0.001
        Q = np.eye(self.num_kps * self.dim_state) * transition_covariance_noise
        self.kf.Q = Q

        # Set up the measurement noise covariance matrix
        observation_covariance_noise = 10
        R = np.eye(2 * self.num_kps) * observation_covariance_noise
        self.kf.R = R

        # Initialize the state vector and covariance matrix
        initial_state = np.zeros(self.num_kps * self.dim_state)
        initial_covariance = np.eye(self.num_kps * self.dim_state)
        self.kf.x = initial_state
        self.kf.x[: self.num_kps * self.dim_measure] = self._state_to_z(a_state=a_state)
        self.kf.P = initial_covariance

    def _state_to_z(self, a_state: State):
        return a_state.pose.to_xy().flatten()

    def _x_to_state(self):
        scores = []
        for kp in self.states[-1].pose.items:
            scores.append(kp.score)

        kps = []
        for i, score in zip(range(self.num_kps), scores):
            idx = 4 * i
            kps.append([int(self.kf.x[idx]), int(self.kf.x[idx + 1]), score])

        pose = Pose2D.from_xy(kps)
        return pose

    def update(self, a_state: State):
        if a_state is not None:
            self.update_age += 1
            self.states.append(a_state)
            self.kf.update(self._state_to_z(a_state=a_state))
        else:
            self.kf.update(None)

    def predict(self):
        self.kf.predict()
        return self._x_to_state()