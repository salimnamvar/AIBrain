"""Machine Learning - Object Tracking - OCSORT Target Utilities

This module provides utilities for working with OCSORT targets.

Classes:
    Target: Represents a target being tracked by the OCSORT algorithm.
    TargetDict: A dictionary-like container for managing multiple OCSORT targets.
    TargetNestedDict: A nested dictionary-like container for managing multiple OCSORT target dictionaries.

Type Variables:
    _BT: A type variable representing the bounding box type for the target.
    _TT: A type variable representing the target type for the target.

Type Aliases:
    AnyBox: A type alias for any bounding box type.
    IntBox: A type alias for integer-based bounding box types.
    FloatBox: A type alias for float-based bounding box types.
    AnyTarget: A type alias for any target type.
    IntTarget: A type alias for integer-based target types.
    FloatTarget: A type alias for float-based target types.
"""

from copy import deepcopy
from dataclasses import asdict, astuple, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Self, TypeAlias, TypeVar, Union, cast

import numpy as np
from filterpy.kalman import KalmanFilter

from src.utils.cnt.b_dict import BaseDict
from src.utils.cv.geom.box import AnyBox, FloatBox
from src.utils.cv.geom.box.bbox2d import BBox2D, FloatBBox2D, IntBBox2D
from src.utils.cv.geom.box.box2d import Box2D, FloatBox2D, IntBox2D
from src.utils.cv.geom.box.sbbox2d import FloatSegBBox2D, IntSegBBox2D, SegBBox2D
from src.utils.ml.trk.ocsort.utils.obs import Observations
from src.utils.ml.trk.ocsort.utils.stats import Stats

if TYPE_CHECKING:
    AnyTarget: TypeAlias = Union[
        "Target[IntBox2D]",
        "Target[FloatBox2D]",
        "Target[IntBBox2D]",
        "Target[FloatBBox2D]",
        "Target[IntSegBBox2D]",
        "Target[FloatSegBBox2D]",
    ]
    IntTarget: TypeAlias = Union["Target[IntBox2D]", "Target[IntBBox2D]", "Target[IntSegBBox2D]"]
    FloatTarget: TypeAlias = Union["Target[FloatBox2D]", "Target[FloatBBox2D]", "Target[FloatSegBBox2D]"]
else:
    AnyTarget = Union[
        "Target[IntBox2D]",
        "Target[FloatBox2D]",
        "Target[IntBBox2D]",
        "Target[FloatBBox2D]",
        "Target[IntSegBBox2D]",
        "Target[FloatSegBBox2D]",
    ]
    IntTarget = Union["Target[IntBox2D]", "Target[IntBBox2D]", "Target[IntSegBBox2D]"]
    FloatTarget = Union["Target[FloatBox2D]", "Target[FloatBBox2D]", "Target[FloatSegBBox2D]"]

BoxT = TypeVar("BoxT", bound=AnyBox, default=FloatBox)
TargetT = TypeVar("TargetT", bound=AnyTarget, default=FloatTarget)


@dataclass
class Target(Generic[BoxT]):
    """OCSORT Target Data Class

    This class represents a target being tracked by the OCSORT algorithm.

    Attributes:
        id (int): Unique identifier for the target.
        obs (Observations[int, BoxT]): Observation history of the target.
        stats (Stats): Statistics for the target.
        _kf (KalmanFilter): Kalman filter for state estimation.
    """

    id: int = field(compare=True, metadata={"description": "Unique identifier for the target."})
    obs: Observations[BoxT] = field(compare=False, metadata={"description": "Observation history of the target."})
    stats: Stats = field(compare=False, metadata={"description": "Statistics for the target."})
    _kf: KalmanFilter = field(
        default_factory=lambda: KalmanFilter(dim_x=7, dim_z=4),
        init=False,
        repr=False,
        compare=False,
        metadata={"description": "Kalman filter for state estimation."},
    )

    def __post_init__(self) -> None:
        """Post-Initialization of the Target instance.

        This method initializes the Kalman filter with the target's initial observation.
        It sets up the state transition matrix, observation matrix, process noise covariance,
        measurement noise covariance, and initial state estimate.

        Raises:
            ValueError: If the observation history is empty.
        """
        if len(self.obs) == 0:
            raise ValueError("Observation history cannot be empty.")

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
        self._kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self._kf.R[2:, 2:] *= 10.0
        self._kf.P[4:, 4:] *= 1000.0
        self._kf.P *= 10.0
        self._kf.Q[-1, -1] *= 0.01
        self._kf.Q[4:, 4:] *= 0.01
        self._kf.x[:4] = self.obs[-1].to_cxyar()[:4, np.newaxis]

    @classmethod
    def create(
        cls,
        a_id: int,
        a_box: AnyBox,
        a_obs_size: int,
        a_step_timestamp: Optional[float] = None,
        a_step_id: Optional[int] = None,
    ) -> Self:
        """Create a new Target instance.

        Args:
            a_step_timestamp (float): Timestamp of the target's birth.
            a_step_id (int): Step/frame number of the target's birth.
            a_id (int): Unique identifier for the target.
            a_box (AnyBox): Initial bounding box of the target.
            a_obs_size (int): Maximum size of the observation history.

        Returns:
            Self: A new instance of Target with initialized observations and statistics.
        """
        stats = Stats(
            birth_timestamp=a_step_timestamp,
            birth_step=a_step_id,
        )
        observations: Observations[BoxT] = Observations[BoxT](a_max_size=a_obs_size)
        observations[stats.predict_count] = cast(BoxT, a_box)
        return cls(id=a_id, obs=observations, stats=stats)

    def _update_velocity(self, a_obs_lookback_step: int, a_box: Optional[AnyBox] = None) -> None:
        """Update the target's velocity based on the observation history.

        Args:
            a_obs_lookback_step (int): The number of steps to look back for the observation.
            a_box (Optional[AnyBox]): The new observation bounding box.
        """
        if a_box is not None:
            if len(self.obs) > 0:
                prev_box = self.obs[-1]
                target_age = self.stats.predict_count - a_obs_lookback_step
                for step_offset in range(a_obs_lookback_step):
                    candidate_age = target_age + step_offset
                    if candidate_age in self.obs:
                        prev_box = self.obs[candidate_age]
                        break
                self.stats.velocity = prev_box.velocity(a_box=a_box)

    def update(self, a_obs_lookback_step: int, a_box: Optional[AnyBox] = None) -> None:
        """Update the target's state with the new observation.

        Args:
            a_obs_lookback_step (int): The number of steps to look back for the observation.
            a_box (Optional[AnyBox]): The new observation bounding box.
        """
        if a_box is not None:
            self._update_velocity(a_obs_lookback_step=a_obs_lookback_step, a_box=a_box)
            self.obs[self.stats.predict_count] = cast(BoxT, a_box)
            self.stats.update()
            self._kf.update(a_box.to_cxyar()[:4, np.newaxis])
        else:
            self._kf.update(None)

    def _prevent_negative_area(self) -> None:
        """Prevent negative area in the bounding box by adjusting the Kalman filter state."""
        if (self._kf.x[6] + self._kf.x[2]) <= 0:
            self._kf.x[6] *= 0.0

    def get_state(self) -> BoxT | None:
        """Get the current state of the target.

        Returns:
            _VT | None: The current state of the target or None if invalid.
        """
        coords = self._kf.x.squeeze()
        if np.any(coords[[2, 3]] < 0):
            return None

        last_box = self.obs[-1]
        match last_box:
            case BBox2D():
                box = BBox2D.from_cxyar(
                    a_coords=coords[:4], a_use_float=True, a_label=last_box.label, a_score=last_box.score
                )
            case SegBBox2D():
                box = SegBBox2D.from_cxyar(
                    a_coords=coords[:4],
                    a_use_float=True,
                    a_label=last_box.label,
                    a_score=last_box.score,
                    a_mask=last_box.mask.data,
                )
            case Box2D():
                box = Box2D.from_cxyar(a_coords=coords, a_use_float=True)
            case _:
                box = Box2D.from_cxyar(a_coords=coords, a_use_float=True)
        return cast(BoxT, box)

    def predict(self) -> BoxT | None:
        """Predict the next state of the target.

        Returns:
            BoxT | None: The predicted state of the target.
        """
        self._prevent_negative_area()
        self._kf.predict()
        self.stats.predict()
        box = self.get_state()
        return cast(BoxT, box)

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


class TargetDict(BaseDict[int, TargetT]):
    """Target Dictionary Data Container

    Attributes:
        data (Dict[int, TargetT]): The Target data contained in the TargetDict.
    """

    def __init__(
        self, a_dict: Dict[int, TargetT] | None = None, a_max_size: int | None = None, a_name: str = "TargetDict"
    ):
        """Initialize the TargetDict.

        Args:
            a_dict (Dict[int, TargetT] | None): The initial dictionary to populate the TargetDict.
            a_max_size (int | None): The maximum size of the TargetDict.
            a_name (str): The name of the TargetDict.
        """
        super().__init__(a_dict=a_dict, a_max_size=a_max_size, a_name=a_name)
        self._id_count = -1

    def get_new_id(self) -> int:
        """Return a new unique ID based on the current maximum key plus one.

        Returns:
            int: The next available unique ID.
        """
        self._id_count += 1
        return self._id_count


class TargetNestedDict(BaseDict[int, TargetDict[TargetT]]):
    """Target Nested Dictionary Data Container

    Attributes:
        data (Dict[int, TargetDict[TargetT]]): The TargetDict data contained in the TargetNestedDict.
    """

    def __init__(
        self,
        a_dict: Dict[int, TargetDict[TargetT]] | None = None,
        a_max_size: int | None = None,
        a_name: str = "TargetNestedDict",
    ):
        """Initialize the TargetNestedDict.

        Args:
            a_dict (Dict[int, TargetDict[TargetT]] | None): The initial dictionary to populate the TargetNestedDict.
            a_max_size (int | None): The maximum size of the TargetNestedDict.
            a_name (str): The name of the TargetNestedDict.
        """
        super().__init__(a_dict=a_dict, a_max_size=a_max_size, a_name=a_name)
