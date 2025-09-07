"""Machine Learning - Object Tracking - OCSORT Statistics Utilities

This module provides a data class for managing statistics related to the OCSORT object tracking algorithm.

Classes:
    Stats: A data class that holds various statistics related to the tracking of targets in the OCSORT algorithm.
"""

from copy import deepcopy
from dataclasses import asdict, astuple, dataclass, field
from typing import Any, Optional, Self, Tuple


@dataclass
class Stats:
    """Statistics for OCSORT Target Tracking

    This class holds various statistics related to the tracking of targets in the OCSORT algorithm.

    Attributes:
        birth_timestamp (float): Timestamp when the target was created.
        birth_step (int): Step/frame number when the target was created.
        total_age (int): Total age counter that increments on each update or prediction.
        predict_count (int): Cumulative count of predictions made, never resets.
        update_count (int): Cumulative count of updates made, never resets.
        age_since_update (int): Number of predictions since the last update, resets on update.
        age_since_predict (int): Number of updates since the last prediction, resets on prediction.
        velocity (Tuple[float, float]): Target velocity in (x, y) coordinates, updated during updates only.
    """

    total_age: int = field(
        default=0, metadata={"description": "Total age counter (increments on update/predict)."}, compare=False
    )
    predict_count: int = field(
        default=0, metadata={"description": "Cumulative prediction counter (never resets)."}, compare=False
    )
    update_count: int = field(
        default=0, metadata={"description": "Cumulative update counter (never resets)."}, compare=False
    )
    age_since_update: int = field(
        default=0, metadata={"description": "Predictions since last update (resets on update)."}, compare=False
    )
    age_since_predict: int = field(
        default=0, metadata={"description": "Updates since last prediction (resets on prediction)."}, compare=False
    )
    velocity: Tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={"description": "Target velocity (x, y) updated during updates only."},
        compare=False,
    )
    birth_timestamp: Optional[float] = field(
        default=None, metadata={"description": "Timestamp when the target was created."}, compare=False
    )
    birth_step: Optional[int] = field(
        default=None, metadata={"description": "Step when the target was created."}, compare=False
    )

    def to_dict(self) -> dict[str, object]:
        """Convert the data class to a dictionary.

        Returns:
            dict[str, object]: Dictionary representation of the data class.
        """
        return asdict(self)

    def to_tuple(self) -> tuple[Any, ...]:
        """Convert the data class to a tuple.

        Returns:
            tuple[Any, ...]: Tuple representation of the data class.
        """
        return astuple(self)

    def to_list(self) -> list[Any]:
        """Convert the data class to a list.

        Returns:
            list[Any]: List representation of the data class.
        """
        return list(self.to_tuple())

    def copy(self) -> Self:
        """Create a deep copy of the data class.

        Returns:
            Self: A new instance of the data class with the same data.
        """
        return deepcopy(self)

    def update(self) -> None:
        """Update Statistics

        Increments the age and update counters, resets post-update prediction age.
        """
        self.total_age += 1
        self.update_count += 1
        self.age_since_update = 0
        self.age_since_predict += 1

    def predict(self) -> None:
        """Predict Statistics

        Increments the age and prediction counters, resets post-prediction update age.
        """
        self.total_age += 1
        self.predict_count += 1
        if self.age_since_update > 0:
            self.age_since_predict = 0
        self.age_since_update += 1
