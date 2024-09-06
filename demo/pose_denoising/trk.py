"""Pose Tracker Module

"""

# region Imported Dependencies
from typing import List

import numpy as np
from filterpy.kalman import KalmanFilter

from brain.cv.shape.ps import Pose2D
from brain.ml.seg import SegBBox2D
from brain.obj import BaseObject, BaseObjectList
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
        #F2 = np.eye(self.num_kps * self.dim_state)
        for i in range(self.num_kps):
            idx = 4 * i
            F[idx : idx + 2, idx + 2 : idx + 4] = np.eye(2)  # position to velocity

        #    F2[i * 4, i * 4 + 2] = 1  # x
        #    F2[i * 4 + 1, i * 4 + 3] = 1  # y

        self.kf.F = F

        # Define the measurement matrix
        H = np.zeros((2 * self.num_kps, self.num_kps * self.dim_state))
        #H2 = np.zeros((2 * self.num_kps, self.num_kps * self.dim_state))
        for i in range(self.num_kps):
            idx = 4 * i
            H[2 * i : 2 * i + 2, idx : idx + 2] = np.eye(2)  # x, y positions

        #    H2[i * 2, i * 4] = 1  # x
        #    H2[i * 2 + 1, i * 4 + 1] = 1  # y
        self.kf.H = H

        # Set up the process noise covariance matrix
        process_noise = 0.001
        Q = np.eye(self.num_kps * self.dim_state) * process_noise
        #Q2 = np.eye(self.num_kps * self.dim_state) * process_noise
        #for i in range(self.num_kps):
        #    Q2[i * 4, i * 4] = process_noise  # Variance for x position
        #    Q2[i * 4 + 1, i * 4 + 1] = process_noise  # Variance for y position
        #    Q2[i * 4 + 2, i * 4 + 2] = process_noise  # Variance for x velocity
        #    Q2[i * 4 + 3, i * 4 + 3] = process_noise  # Variance for y velocity
        self.kf.Q = Q

        # Set up the measurement noise covariance matrix
        measurement_noise = 10
        R = np.eye(2 * self.num_kps) * measurement_noise
        #R2 = np.eye(2 * self.num_kps) * measurement_noise
        #for i in range(self.num_kps):
        #    R2[i * 2, i * 2] = measurement_noise  # Variance for x position measurement
        #    R2[i * 2 + 1, i * 2 + 1] = measurement_noise  # Variance for y position measurement
        self.kf.R = R

        # Covariance matrix
        initial_uncertainty = 100
        P = np.eye(self.num_kps * self.dim_state)

        #P2 = np.eye(self.num_kps * self.dim_state) * initial_uncertainty
        #for i in range(self.num_kps):
        #    P2[i * 4, i * 4] = initial_uncertainty  # Variance for x position
        #    P2[i * 4 + 1, i * 4 + 1] = initial_uncertainty  # Variance for y position
        #    P2[i * 4 + 2, i * 4 + 2] = initial_uncertainty  # Variance for x velocity
        #    P2[i * 4 + 3, i * 4 + 3] = initial_uncertainty  # Variance for y velocity
        self.kf.P = P

        # Measurement
        #initial_state = np.zeros(self.num_kps * self.dim_state)
        #self.kf.x = initial_state
        #self.kf.x[: self.num_kps * self.dim_measure] = self._state_to_z(a_state=a_state)

        measurement = a_state.pose.to_xy().flatten()
        reshaped_array = np.zeros(2 * self.num_kps * 2)

        for i in range(self.num_kps):
            x = measurement[2 * i]
            y = measurement[2 * i + 1]
            reshaped_array[4 * i] = x
            reshaped_array[4 * i + 1] = y

            # Assuming initial velocities are zero; you may update with actual velocities if available
            reshaped_array[4 * i + 2] = 0.0  # x' velocity
            reshaped_array[4 * i + 3] = 0.0  # y' velocity

        #initial_x_positions = [0.0] * self.num_kps  # Replace with actual initial positions if known
        #initial_y_positions = [0.0] * self.num_kps  # Replace with actual initial positions if known
        #initial_x_velocities = [0.0] * self.num_kps  # Replace with actual initial velocities if known
        #initial_y_velocities = [0.0] * self.num_kps  # Replace with actual initial velocities if known

        #initial_state = []
        #for i in range(self.num_kps):
        #    initial_state.extend([initial_x_positions[i], initial_y_positions[i],
        #                          initial_x_velocities[i], initial_y_velocities[i]])

        self.kf.x = reshaped_array#np.array(initial_state).reshape(-1, 1)

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

    def state(self):
        return self._x_to_state()