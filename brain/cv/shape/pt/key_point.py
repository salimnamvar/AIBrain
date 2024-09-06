""" KeyPoint Module

This module defines a class for representing 2D keypoints and a container class for a collection of such keypoints.
"""

# region Import Dependencies
from typing import Union, List, Optional, Tuple

import numpy as np

from brain.cv.shape.pt import Point2D, Point2DList
from brain.misc import is_float
from brain.obj import BaseObjectList


# endregion Import Dependencies


class KeyPoint2D(Point2D):
    """KeyPoint2D

    This class represents a 2D keypoint with x, y coordinates and an optional score.

    Attributes:
        x (Union[int, float]):
            The x-coordinate of the keypoint.
        y (Union[int, float]):
            The y-coordinate of the keypoint.
        score (float, optional):
            The score associated with the keypoint, indicating its confidence level.
    """

    def __init__(
        self,
        a_x: Union[int, float],
        a_y: Union[int, float],
        a_score: Optional[float] = None,
        a_name: str = "KEY_POINT",
    ) -> None:
        """Initialize KeyPoint2D

        This constructor initializes a KeyPoint2D instance with the provided x and y coordinates, an optional score,
        and a name.

        Args:
            a_x (Union[int, float]):
                The x-coordinate of the keypoint.
            a_y (Union[int, float]):
                The y-coordinate of the keypoint.
            a_score (Optional[float], optional):
                The score associated with the keypoint, indicating its confidence level (default is None).
            a_name (str, optional):
                A string specifying the name of the keypoint (default is 'KEY_POINT').

        Returns:
            None
        """
        # Validate inputs
        x, y, score = self._validate(a_x=a_x, a_y=a_y, a_score=a_score)

        # Initialize keypoint
        super().__init__(a_x=x, a_y=y, a_name=a_name)
        self.score: float = a_score

    def _validate_dtypes(
        self,
        a_x: Union[int, float],
        a_y: Union[int, float],
        a_score: Optional[float] = None,
    ) -> Tuple[int, int, float]:
        """Validate Data Types for KeyPoint2D

        This private method validates the data types of the keypoint coordinates (x, y) and the optional score.

        Args:
            a_x (Union[int, float]):
                The x-coordinate of the keypoint.
            a_y (Union[int, float]):
                The y-coordinate of the keypoint.
            a_score (Optional[float], optional):
                The score associated with the keypoint, indicating its confidence level (default is None).

        Returns:
            Tuple[int, int, float]:
                A tuple containing the validated x-coordinate, y-coordinate, and score (if provided).

        """
        # Convert the data types of points to be int
        a_x = int(a_x)
        a_y = int(a_y)

        # Convert score data type to be a float
        if a_score is not None:
            a_score = float(a_score)

        return a_x, a_y, a_score

    def _validate(
        self,
        a_x: Union[int, float],
        a_y: Union[int, float],
        a_score: Optional[float] = None,
    ) -> Tuple[int, int, float]:
        """Validate KeyPoint2D Coordinates

        This private method corrects the data types of the keypoint coordinates (x, y) and the optional score.

        Args:
            a_x (Union[int, float]):
                The x-coordinate of the keypoint.
            a_y (Union[int, float]):
                The y-coordinate of the keypoint.
            a_score (Optional[float], optional):
                The score associated with the keypoint, indicating its confidence level (default is None).

        Returns:
            Tuple[int, int, float]:
                A tuple containing the corrected x-coordinate, y-coordinate, and score (if provided).

        Note:
            The validation process can be expanded based on the use-case.

        """
        # Correct data types
        x, y, score = self._validate_dtypes(a_x, a_y, a_score)

        # NOTE: The validation process can be expanded based on the use-case
        return x, y, score

    @property
    def score(self) -> float:
        """Get the score of the keypoint.

        Returns:
            float:
                The score associated with the keypoint.

        """
        return self._score

    @score.setter
    def score(self, a_score: float = None):
        """Set the score of the keypoint.

        Args:
            a_score (float, optional):
                The score associated with the keypoint, indicating its confidence level (default is None).

        Raises:
            TypeError: If the provided score is not a float.
        """
        if a_score is not None:
            float_flag = is_float(a_score)
            if not float_flag:
                raise TypeError("The `a_score` should be a float.")
            if float_flag:
                a_score = float(a_score)
                if a_score > 1.0:
                    a_score = a_score / 100.0
        self._score: float = a_score

    def to_dict(self) -> dict:
        """Convert the keypoint to a dictionary.

        Returns:
            dict:
                A dictionary representation of the keypoint, including its x and y coordinates and score.
        """
        dic = {"x": self.x, "y": self.y, "score": self.score}
        return dic

    @classmethod
    def from_xys(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "KeyPoint2D":
        """Create a KeyPoint2D instance from (x, y, s) coordinates.

        This method creates a KeyPoint2D instance based on the provided XY coordinates and additional keyword arguments.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                The XY coordinates. For a simple keypoint, pass a tuple or list with two elements (x, y, s).
                For a keypoint with a score, pass a tuple or list with three elements (x, y, score).
            **kwargs:
                Additional keyword arguments to pass to the KeyPoint2D constructor.

        Returns:
            KeyPoint2D: A KeyPoint2D instance.

        Raises:
            ValueError:
                If the number of coordinates is not 2 or 3.
        """
        cls.validate_array(a_coordinates=a_coordinates)

        if len(a_coordinates) == 2:
            keypoint: KeyPoint2D = KeyPoint2D(a_x=a_coordinates[0], a_y=a_coordinates[1], **kwargs)
        elif len(a_coordinates) == 3:
            keypoint: KeyPoint2D = KeyPoint2D(
                a_x=a_coordinates[0],
                a_y=a_coordinates[1],
                a_score=a_coordinates[2],
                **kwargs,
            )
        else:
            raise ValueError("Invalid number of coordinates for keypoint.")

        return keypoint

    def to_xys(self) -> np.ndarray:
        """Convert to NumPy Array of format [x, y, s]

        Returns:
            np.ndarray: A NumPy array representation of the keypoint.
        """
        return np.concatenate((self.to_xy(), [self.score]))


class KeyPoint2DList(Point2DList, BaseObjectList[KeyPoint2D]):
    """A list of 2D keypoints.

    This class extends the BaseObjectList to specifically handle lists of 2D keypoints.

    Attributes:
        name (str): A string specifying the name of the KeyPoint2DList instance.
        max_size (int): An integer representing the maximum size of the list.
        items (List[KeyPoint2D]): A list of KeyPoint2D instances contained within the KeyPoint2DList.
    """

    def __init__(
        self,
        a_name: str = "KeyPoint2DList",
        a_max_size: int = -1,
        a_items: List[KeyPoint2D] = None,
    ):
        """Initialize a KeyPoint2DList.

        Args:
            a_name (str, optional):
                A string specifying the name of the KeyPoint2DList instance (default is 'KeyPoint2DList').
            a_max_size (int, optional):
                An integer representing the maximum size of the list (default is -1, indicating no size limit).
            a_items (List[KeyPoint2D], optional):
                A list of KeyPoint2D instances to initialize the KeyPoint2DList (default is None).

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    @classmethod
    def from_xys(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "KeyPoint2DList":
        """Create a KeyPoint2DList from a list of (x, y, s) coordinates.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): The input coordinates to create KeyPoint2D instances.
            **kwargs: Additional keyword arguments to pass to the KeyPoint2D constructor.

        Returns:
            KeyPoint2DList: A KeyPoint2DList containing KeyPoint2D instances created from the input coordinates.

        Raises:
            TypeError: If `a_coordinates` is None or not an instance of Tuple, List, or np.ndarray.
            ValueError: If the shape of the array is less than 2.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate points
        keypoints = cls()
        keypoints.append(a_item=[KeyPoint2D.from_xys(a_coordinates=coord, **kwargs) for coord in coordinates])
        return keypoints

    def to_xys(self) -> np.ndarray:
        """Convert the KeyPoint2DList to a NumPy array of (x, y, s) coordinates.

        Returns:
            np.ndarray: A NumPy array containing (x, y, s) coordinates of all keypoints in the KeyPoint2DList.
        """
        if len(self.items):
            arr = np.vstack([point.to_xys() for point in self.items])
        else:
            arr = np.empty(shape=(0, 3))
        return arr
