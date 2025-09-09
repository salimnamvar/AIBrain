"""Machine Learning - Object Tracking - OCSORT Observation Utils

This module provides utilities for managing and retrieving historical observations
in the context of the OCSORT object tracking algorithm. It defines a dictionary-like
structure to store observations and includes methods for accessing previous states
for temporal consistency in tracking.

Classes:
    Observations: A dictionary to store observations with methods to retrieve previous states.

Type Variables:
    _VT:
        Value type constrained to various 2D geometric objects (Box2D, BBox2D, SegBBox2D)
        with Point2D coordinates supporting both integer and float precision
"""

from typing import Dict, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from aib.cnt.b_dict import BaseDict
from aib.cv.geom.box import AnyBox, FloatBox

BoxT = TypeVar("BoxT", bound=AnyBox, default=FloatBox)


class Observations(BaseDict[int, BoxT]):
    """Observation History Dictionary for OCSORT Tracking

    This class extends BaseDict to manage a history of observations indexed by age (int).
    It provides methods to retrieve previous observations for temporal consistency in tracking.

    Attributes:
        data (Optional[Dict[int, BBox2D[Point2D[int]] | BBox2D[Point2D[float]]]]):
            Dictionary of observations with integer keys (ages) and BBox2D values.
    """

    def __init__(
        self,
        a_dict: Optional[Dict[int, BoxT]] = None,
        a_max_size: int | None = None,
        a_name: str = "Observations",
    ):
        """
        Initialize the observation history dictionary.

        Args:
            a_dict (Optional[Dict[int, BBox2D[Point2D[int]] | BBox2D[Point2D[float]]]]):
                Initial dictionary of observations with integer keys (ages) and BBox2D values.
            a_max_size (int | None):
                Maximum size of the dictionary. If None, no limit is enforced.
            a_name (str):
                Name of the dictionary for identification purposes.
        """
        super().__init__(a_dict=a_dict, a_max_size=a_max_size, a_name=a_name)

    def get_prev_obs(self, a_curr_age: int, a_lookback_steps: int) -> npt.NDArray[np.floating | np.integer]:
        """
        Retrieve historical observation from tracker's observation history for temporal consistency.

        This function implements temporal lookback in OC-SORT tracking, finding the k-th previous
        observation state to compute velocity direction consistency (VDC) and motion prediction.

        Args:
            a_curr_age (int):
                Current age/frame number of the tracker (>=0) used as reference point for temporal lookback.
            a_lookback_steps (int):
                Number of time steps to look back (k in temporal window). Typically 1-5 for short-term motion estimation.

        Returns:
            npt.NDArray[np.floating | np.integer]:
                Historical bounding box in format [x1, y1, x2, y2, score] and shape of (5,) - dtype=float64.
                Returns [-1, -1, -1, -1, -1] if no valid observation found.

        Algorithm:
            1. Search for exact age match: (current_age - lookback_steps)
            2. If not found, try progressively closer time steps: (current_age - lookback_steps + i)
            3. Fallback to most recent available observation if no exact match
            4. Return invalid marker [-1, -1, -1, -1, -1] if no observations exist

        Note:
            - Used in VDC calculation for motion consistency validation
            - Temporal window should balance motion estimation accuracy vs computational cost
            - Invalid observations marked with negative coordinates for easy detection
        """
        invalid_obs = np.array(
            [
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            dtype=np.float32,
        )

        if len(self) == 0:
            return invalid_obs

        target_age = a_curr_age - a_lookback_steps

        for step_offset in range(a_lookback_steps):
            candidate_age = target_age + step_offset
            if candidate_age in self:
                return self.get_by_key(candidate_age).to_xyxy()[:5]

        most_recent_age = max(self.keys())
        return self.get_by_key(most_recent_age).to_xyxy()[:5]
