"""Target Statistics Module.

This module defines the `TargetStatistics` class for handling statistics associated with a trackable target.

"""

# region Imported Dependencies
from typing import Tuple
from brain.util.obj import BaseObject

# endregion Imported Dependencies


class TargetStatistics(BaseObject):
    """Target Statistics

        Class for handling statistics associated with a trackable target.

    Attributes:
        age (int):
            Age of the target that increases by any update or predict status.
        prediction_age (int):
            Age of the target in the prediction mode. This counter increases over predictions and does not reset.
        update_age (int):
            Age of the target in the update mode. This counter increases over updates and does not reset.
        post_update_prediction_age (int):
            Prediction age of the target after the latest update condition. Increases in the prediction state and
            resets to 0 on the update state.
        post_prediction_update_age (int):
            Update age of the target after the latest prediction condition. Increases in the update state and resets
            to 0 on the prediction state.
        velocity (Tuple[float, float]):
            Speed of the target in the x and y directions. This is updated during update states, not predictions.
    """

    def __init__(self, a_name: str = "Target_Statistics"):
        """Initialize a TargetStatistics instance.

        Args:
            a_name (str, optional): Name of the target statistics (default is "Target_Statistics").

        Raises:
            TypeError: If input parameters are of incorrect types.
        """
        super().__init__(a_name)
        self.age: int = 0
        self.prediction_age: int = 0
        self.update_age: int = 0
        self.post_update_prediction_age: int = 0
        self.post_prediction_update_age: int = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        """Convert the TargetStatistics instance to a dictionary.

        Returns:
            dict: A dictionary representation of the TargetStatistics instance, including its attributes.
        """
        dic = {
            "name": self.name,
            "age": self.age,
            "prediction_age": self.prediction_age,
            "update_age": self.prediction_age,
            "post_update_prediction_age": self.post_update_prediction_age,
            "post_prediction_update_age": self.post_prediction_update_age,
            "velocity": self.velocity,
        }
        return dic

    @property
    def age(self) -> int:
        """Get the age of the target in the both prediction and update mode.

        Returns:
            int: The age of the target.
        """
        return self._age

    @age.setter
    def age(self, a_age: int) -> None:
        """Set the age of the target in the both prediction and update mode.

        Args:
            a_age (int): Age of the target.

        Raises:
            TypeError: If the input age is not an instance of `int`.
        """
        if a_age is None or not isinstance(a_age, int):
            raise TypeError("The `a_age` must be a `int`.")
        self._age: int = a_age

    @property
    def prediction_age(self) -> int:
        """Get the age of the target in the prediction mode.

        Returns:
            int: The age of the target in the prediction mode.
        """
        return self._prediction_age

    @prediction_age.setter
    def prediction_age(self, a_age: int) -> None:
        """Set the age of the target in the prediction mode.

        Args:
            a_age (int): Age of the target in the prediction mode.

        Raises:
            TypeError: If the input age is not an instance of `int`.
        """
        if a_age is None or not isinstance(a_age, int):
            raise TypeError("The `a_age` must be a `int`.")
        self._prediction_age: int = a_age

    @property
    def update_age(self) -> int:
        """Get the age of the target in the update mode.

        Returns:
            int: The age of the target in the update mode.
        """
        return self._update_age

    @update_age.setter
    def update_age(self, a_age: int) -> None:
        """Set the age of the target in the update mode.

        Args:
            a_age (int): Age of the target in the update mode.

        Raises:
            TypeError: If the input age is not an instance of `int`.
        """
        if a_age is None or not isinstance(a_age, int):
            raise TypeError("The `a_age` must be a `int`.")
        self._update_age: int = a_age

    @property
    def post_update_prediction_age(self) -> int:
        """Get the prediction age of the target after the latest update condition.

        Returns:
            int: The prediction age of the target after the latest update condition.
        """
        return self._post_update_prediction_age

    @post_update_prediction_age.setter
    def post_update_prediction_age(self, a_age: int) -> None:
        """Set the prediction age of the target after the latest update condition.

        Args:
            a_age (int): Prediction age of the target after the latest update condition.

        Raises:
            TypeError: If the input age is not an instance of `int`.
        """
        if a_age is None or not isinstance(a_age, int):
            raise TypeError("The `a_age` must be a `int`.")
        self._post_update_prediction_age: int = a_age

    @property
    def post_prediction_update_age(self) -> int:
        """Get the update age of the target after the latest prediction condition.

        Returns:
            int: The update age of the target after the latest prediction condition.
        """
        return self._post_prediction_update_age

    @post_prediction_update_age.setter
    def post_prediction_update_age(self, a_age: int) -> None:
        """Set the update age of the target after the latest prediction condition.

        Args:
            a_age (int): Update age of the target after the latest prediction condition.

        Raises:
            TypeError: If the input age is not an instance of `int`.
        """
        if a_age is None or not isinstance(a_age, int):
            raise TypeError("The `a_age` must be a `int`.")
        self._post_prediction_update_age: int = a_age

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get the speed of the trackable target in the x and y directions.

        Returns:
            Tuple[float, float]: The speed of the trackable target in the x and y directions.
        """
        return self._velocity

    @velocity.setter
    def velocity(self, a_velocity: Tuple[float, float]) -> None:
        """Set the speed of the trackable target in the x and y directions.

        Args:
            a_velocity (Tuple[float, float]): Speed of the trackable target in the x and y directions.

        Raises:
            TypeError: If the input velocity is not an instance of `Tuple[float, float]`.
        """
        if a_velocity is None or not all(
            [isinstance(val, float) for val in a_velocity]
        ):
            raise TypeError("The `a_velocity` must be a `Tuple[float, float]`.")
        self._velocity: Tuple[float, float] = a_velocity

    def update_stats(self, a_mode: str) -> None:
        """Update age statistics based on the provided mode.

        Args:
            a_mode (str): The mode to update age statistics. Should be either 'update' or 'predict'.

        Raises:
            TypeError: If the input mode is not a string or not one of 'update' or 'predict'.
        """
        if (
            a_mode is None
            or not isinstance(a_mode, str)
            and a_mode.lower() not in ["update", "predict"]
        ):
            raise TypeError(
                "The `a_mode` must be a `str` and be `update` or `predict`."
            )

        a_mode: str = a_mode.lower()
        if a_mode == "update":
            self._adjust_update_age()
        elif a_mode == "predict":
            self._adjust_prediction_age()
        else:
            raise TypeError("The `a_mode` be `update` or `predict`.")

    def _adjust_update_age(self):
        """Adjust age statistics after an update operation.

        - Resets `post_update_prediction_age` to 0.
        - Increments `post_prediction_update_age` by 1.
        """
        self._age += 1
        self._update_age += 1
        self._post_update_prediction_age = 0
        self._post_prediction_update_age += 1

    def _adjust_prediction_age(self):
        """Adjust age statistics after a prediction operation.

        - Increments `prediction_age` by 1.
        - If `post_update_prediction_age` is greater than 0, resets `post_prediction_update_age` to 0.
        - Increments `post_update_prediction_age` by 1.
        """
        self._age += 1
        self._prediction_age += 1
        if self._post_update_prediction_age > 0:
            self._post_prediction_update_age = 0
        self._post_update_prediction_age += 1
