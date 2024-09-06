"""Trackable Kalman-Filter Target.

This module defines the `KFTarget` class, representing a trackable target using the Kalman filter.

"""

# region Imported Dependencies
import uuid
from typing import Optional, List, Union
import numpy as np
from filterpy.kalman import KalmanFilter

from brain.misc import Time
from .stat import TargetStatistics
from .state import State, StateDict
from brain.obj import ExtBaseObject, BaseObjectList, BaseObjectDict
from brain.cv.shape.bx import BBox2D

# endregion Imported Dependencies


class KFTarget(ExtBaseObject):
    """Trackable Kalman-Filter Target.

    This class represents a trackable target using the Kalman filter.

    Attributes:
        id (uuid.UUID): Unique identifier for the target.
        kf (filterpy.kalman.KalmanFilter): Kalman filter used for tracking.
        delta_time (int): Time delay for speed calculation.
        states (StateDict): Dictionary of observation states.
        state (State): Last observation state of the target.
        statistics (TargetStatistics): Statistics associated with the target.

    Note:
        The Kalman Filter's dimensions are followed as:
            State Vector (dim_x = 7):
                - x: X-coordinate of the bounding box center.
                - y: Y-coordinate of the bounding box center.
                - w: Width of the bounding box.
                - h: Height of the bounding box.
                - A: Area of the bounding box.
                - AR: Aspect ratio of the bounding box.
                - t: Time derivative of time.

            Measurement Vector (dim_z = 4):
                - x: Observed X-coordinate of the bounding box center.
                - y: Observed Y-coordinate of the bounding box center.
                - A: Observed area of the bounding box.
                - AR: Observed aspect ratio of the bounding box.
    """

    def __init__(
        self,
        a_state: State,
        a_num_st_thre: int,
        a_time: Time,
        a_delta_time: Optional[int] = 3,
        a_id: Optional[uuid.UUID] = None,
        a_name: Optional[str] = "KFTarget",
    ):
        """Initialize a KFTarget instance

        Args:
            a_state (State): Initial state of the target.
            a_num_st_thre (int): Maximum size of the observation states dictionary.
            a_time (Time): Time information of the initialization step.
            a_delta_time (int, optional): Time delay for speed calculation (default is 3).
            a_id (uuid.UUID, optional): Unique identifier for the target (default is None).
            a_name (str, optional): Name of the target (default is "KFTarget").

        Raises:
            TypeError: If input parameters are of incorrect types.
        """
        if a_state is None or not isinstance(a_state, State):
            raise TypeError("The `a_state` must be a `State`.")
        if a_num_st_thre is None or not isinstance(a_num_st_thre, int):
            raise TypeError("The `a_num_st_thre` must be a `int`.")

        super().__init__(a_name=a_name)
        # Timestamp
        self.time: Time = a_time
        # Unique Identity
        self.id: uuid.UUID = a_id if a_id is not None else uuid.uuid4()
        # Delta time as the delay for calculation of speed
        self.delta_time: int = a_delta_time
        # Statistics
        self._statistics: TargetStatistics = TargetStatistics(a_name=f"{self.name} Statistics")

        # Initialize Observation States
        self._states: StateDict = StateDict(a_max_size=a_num_st_thre, a_name=f"{self.name} States", a_key_type=int)
        # Initialize the latest state
        self._state: State = a_state
        # Add the first state
        self._states[self._statistics.prediction_age] = a_state

        # Target Kalman Filter Tracker
        self._kf = KalmanFilter(dim_x=7, dim_z=4)
        self._init_kf(a_state=a_state)

    def to_dict(self) -> dict:
        """Convert the KFTarget instance to a dictionary.

        Returns:
            dict: A dictionary representation of the KFTarget instance, including its attributes.
        """
        dic = {
            "name": self.name,
            "id": self.id,
            "state": self.state.to_dict(),
            "states": self.states.to_dict(),
            "timestamp": self.time,
            "statistics": self.statistics.to_dict(),
            "kf": self._kf.__repr__(),
            "delta_time": self.delta_time,
        }
        return dic

    def _init_kf(self, a_state: State) -> None:
        """Initialize the Kalman Filter parameters based on the provided state.

        Args:
            a_state (State): The initial state used to configure the Kalman Filter.

        Raises:
            TypeError: If the provided `a_state` is `None` or not an instance of the `State` class.

        The Kalman Filter is configured with the following matrices:
            - State transition matrix (F)
            - Measurement matrix (H)
            - Measurement noise covariance matrix (R)
            - Process noise covariance matrix (P)
            - Process noise matrix (Q)

        The initial state of the Kalman Filter (x) is set based on the provided state's bounding box.

        Note:
            This method is intended for internal use within the KFTarget class.
        """
        if a_state is None and not isinstance(a_state, State):
            raise TypeError("The `a_state` must be a `State`.")

        # State transition matrix
        self._kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # Measurement matrix
        self._kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        # Measurement noise covariance
        self._kf.R[2:, 2:] *= 10.0  # Increase covariance for the measurement noise
        # Give high uncertainty to the unobservable initial velocities
        self._kf.P[4:, 4:] *= 1000.0  # Increase uncertainty for the initial velocities
        self._kf.P *= 10.0  # Increase overall uncertainty
        # Process noise covariance
        self._kf.Q[-1, -1] *= 0.01  # Decrease covariance for process noise
        self._kf.Q[4:, 4:] *= 0.01  # Decrease covariance for process noise

        # Set the initial state using the bounding box in the format of [center_x, center_y, area, aspect_ratio]
        self._kf.x[:4] = a_state.box.to_cxyar()[:, np.newaxis]

    @property
    def id(self) -> uuid.UUID:
        """Get the unique identifier of the Kalman Filter target.

        Returns:
            uuid.UUID: The unique identifier of the target.
        """
        return self._id

    @id.setter
    def id(self, a_id: uuid.UUID) -> None:
        """Set the unique identifier of the Kalman Filter target.

        This method sets the unique identifier attribute of the Kalman Filter target instance.

        Args:
            a_id (uuid.UUID): The new unique identifier value to be set.

        Raises:
            TypeError: If `a_id` is not an instance of the `uuid.UUID` class.

        Returns:
            None
        """
        if a_id is None or not isinstance(a_id, uuid.UUID):
            raise TypeError("The `a_id` must be a `uuid.UUID`.")
        self._id: uuid.UUID = a_id

    @property
    def kf(self):
        """Get the Kalman filter used for tracking.

        Returns:
            filterpy.kalman.KalmanFilter: The Kalman filter instance.
        """
        return self._kf

    @property
    def delta_time(self) -> int:
        """Get or set the time delay for speed calculation.

        Returns:
            int: The time delay for speed calculation.
        """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, a_delta_time: int) -> None:
        """Set the time delay for speed calculation.

        Args:
            a_delta_time (int): The new time delay for speed calculation.

        Raises:
            TypeError: If `a_delta_time` is not an integer.
        """
        if a_delta_time is None or not isinstance(a_delta_time, int):
            raise TypeError("The `a_delta_time` must be a `int`.")
        self._delta_time: int = a_delta_time

    @property
    def states(self) -> StateDict:
        """Get the dictionary of observation states associated with the Kalman Filter target.

        The observation states represent the states of update statuses that are reached just after a prediction.
        It does not include the prediction states, and it excludes update states that are repeated
        immediately after an update state.

        Returns:
            StateDict: The dictionary of observation states.
        """
        return self._states

    @property
    def state(self) -> State:
        """Get the last observation state associated with the Kalman Filter target.

        Returns:
            State: The last observation state.
        """
        return self._state

    @property
    def kf_state(self) -> State:
        """Get the current state of the Kalman Filter target.

        Returns:
            State: The current state of the Kalman Filter target.

        Note:
            This property extracts the state information from the Kalman Filter's internal state vector and
            constructs a new State object with the corresponding bounding box.
        """
        coordinates = self._kf.x.squeeze().copy()
        box = None
        if np.any(coordinates[[2, 3]] < 0):
            state = State(a_box=box)
        else:
            state = State(
                a_box=BBox2D.from_cxyar(
                    a_coordinates=coordinates,
                    a_score=self._state.box.score,
                    a_img_size=self._state.box.img_size,
                    a_strict=self._state.box.strict,
                    a_conf_thre=self._state.box.conf_thre,
                    a_min_size_thre=self._state.box.min_size_thre,
                    a_name=self._state.name,
                    a_do_validate=False,
                )
            )
        return state

    @property
    def statistics(self) -> TargetStatistics:
        """Get the statistics associated with the Kalman Filter target.

        Returns:
            TargetStatistics: The statistics associated with the target, including prediction and update ages.
        """
        return self._statistics

    @statistics.setter
    def statistics(self, a_statistics: TargetStatistics) -> None:
        """Set the statistics associated with the Kalman Filter target.

        This method sets the statistics attribute of the Kalman Filter target instance.

        Args:
            a_statistics (TargetStatistics): The new statistics object to be set.

        Raises:
            TypeError: If `a_statistics` is not an instance of the `TargetStatistics` class.

        Returns:
            None
        """
        if a_statistics is None or not isinstance(a_statistics, TargetStatistics):
            raise TypeError("The `a_statistics` must be a `TargetStatistics`.")
        self._statistics: TargetStatistics = a_statistics

    def _update_speed(self, a_state: State) -> None:
        """Estimate the track speed direction with observations of Delta t steps away.

        Args:
            a_state (State): The current observation state used for speed estimation.

        Notes:
            - If `a_state` is not provided or is `None`, the speed estimation is not performed.
            - The method utilizes the observation states associated with the target to estimate the speed.
            - It considers the observations of Delta t steps away, where Delta t is the time delay for speed calculation.
            - The estimated speed is stored in the target's statistics.

        Returns:
            None
        """
        if a_state is not None:
            if self.state is not None:
                previous_state = None
                for i in range(self.delta_time):
                    delta = self.delta_time - i
                    key = self._statistics.prediction_age - delta
                    if key in self.states:
                        previous_state = self.states[key]
                        break
                if previous_state is None:
                    previous_state = self.state

                self._statistics.velocity = previous_state.box.speed_xy(a_box2d=a_state.box)

    def update(self, a_state: Optional[State] = None):
        """Update the Kalman Filter target with a new observation state.

        Args:
            a_state (Optional[State]): The new observation state to update the target.

        Notes:
            - If `a_state` is not provided or is `None`, only the Kalman Filter's state is updated with `None`.
            - If provided, the method updates the target's speed, last observation state, observation states,
              statistics, Kalman Filter's state.

        Returns:
            None
        """
        if a_state is not None:
            # Update target's speed
            self._update_speed(a_state=a_state)

            # Update target's last observation state
            self._state = a_state

            # Update target's observation states
            self._states[self._statistics.prediction_age] = a_state

            # Update target's statistics
            self._statistics.update_stats(a_mode="update")

            # Update Kalman Filter's state
            self._kf.update(a_state.box.to_cxyar()[:, np.newaxis])

        else:
            # Update Kalman Filter's state as `NONE`
            self._kf.update(None)

    def _adjust_time_derivative(self):
        """
        Adjusts the time derivative of time if the sum of time derivative and area is non-positive.

        This method checks if the sum of the time derivative of time and the area of the bounding box is
        less than or equal to zero. If this condition is met, it sets the time derivative of time to zero.

        Note:
            The adjustment made in this specific case seems to be a mechanism to handle situations where the predicted
            bounding box area becomes non-positive, which may not be physically meaningful or realistic.

        Returns:
            None
        """
        if (self._kf.x[6] + self._kf.x[2]) <= 0:
            self._kf.x[6] *= 0.0

    def predict(self) -> State:
        """Predict the next state of the Kalman Filter target.

        This method performs the prediction step of the Kalman Filter to estimate the next state of the target.

        It involves adjusting the time derivative, predicting the state using the Kalman Filter, updating age
        statistics, and returning the predicted state.

        Returns:
            State: The predicted :class:`State` of the Kalman Filter target.

        Note:
            The prediction updates the internal state of the Kalman Filter target.
        """
        # Adjust time derivative
        self._adjust_time_derivative()

        # Predict state
        self._kf.predict()

        # Update target's statistics
        self._statistics.update_stats(a_mode="predict")

        # Get target's Kalman Filter state
        state = self.kf_state
        return state


class KFTargetList(BaseObjectList[KFTarget]):
    """List container for managing instances of KFTarget.

    This class extends BaseObjectList and is specifically designed for managing a list of KFTarget instances.
    """

    def __init__(
        self,
        a_name: str = "KFTargetList",
        a_max_size: int = -1,
        a_items: List[KFTarget] = None,
    ):
        """Initialize a KFTargetList instance.

        Args:
            a_name (str, optional): Name of the KFTargetList (default is "KFTargetList").
            a_max_size (int, optional): Maximum size limit for the list. If set to -1, there is no limit (default is -1).
            a_items (List[KFTarget], optional): Initial list of KFTarget instances (default is None).

        Returns:
            None
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


class KFTargetDict(BaseObjectDict[uuid.UUID, KFTarget]):
    """Dictionary for storing KFTarget objects with UUID keys.

    This class extends the BaseObjectDict and is specifically designed for storing KFTarget objects
    with UUID keys. It provides additional functionality for managing a dictionary of KFTarget instances.

    Attributes:
        name (str): Name of the KFTargetDict.
        max_size (int): Maximum size limit for the dictionary. If set to -1, there is no limit.
        items (Dict[uuid.UUID, KFTarget]): Dictionary containing KFTarget objects with UUID keys.
    """

    def __init__(
        self,
        a_name: str = "KFTargetDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[KFTarget, List[KFTarget]] = None,
    ):
        """Initialize a KFTargetDict instance.

        Args:
            a_name (str, optional): Name of the KFTargetDict (default is "KFTargetDict").
            a_max_size (int, optional):
                Maximum size limit for the dictionary. If set to -1, there is no limit (default is -1).
            a_key (Union[uuid.UUID, List[uuid.UUID]], optional):
                Initial key or keys of the dictionary (default is None).
            a_value (Union[KFTarget, List[KFTarget]], optional):
                Initial value or values of the dictionary (default is None).

        Returns:
            None
        """
        super().__init__(a_name, a_max_size, a_key, a_value)
