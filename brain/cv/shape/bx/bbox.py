"""Bounding Box Module

This module defines classes for working with 2D bounding boxes, building upon the basic :class:`Box2D` and
:class:`Box2DList` classes.

The primary classes included are:

- :class:`BBox2D`: Represents a 2D bounding box with an additional confidence score.
- :class:`BBox2DList`: A container for a collection of BBox2D objects.

These classes provide functionality for converting bounding boxes to dictionaries, NumPy arrays, and creating instances
from coordinates in the [x, y, x, y, s] format.
"""

# region Import Dependencies
from typing import Union, Tuple, List, Optional

import numpy as np

from brain.cv.shape.pt import Point2D
from brain.cv.shape.sz import Size
from brain.misc import is_float
from brain.obj import BaseObjectList
from .box import Box2D, Box2DList
from .fmt import xywh_to_xyxy, cxyar_to_xyxy, cxywh_to_xyxy
from .status import CoordStatus, ConfidenceStatus, SizeStatus


# endregion Import Dependencies


class BBox2D(Box2D):
    """2D Bounding Box

    This class represents a 2D bounding box with an additional confidence score. It extends the functionality of the
    base :class:`Box2D` class.

    Attributes:
        p1 (Point2D):
            The top-left point of the box as :class:`Point2D`.
        p2 (Point2D):
            The bottom-right point of the box as :class:`Point2D`.
        score (float):
            The confidence score associated with the bounding box.
        img_size (Size):
            The size of the image the bounding box is associated with.
        strict (bool):
            A flag indicating whether to enforce strict conditions for the bounding box.
        coord_status (CoordStatus):
            An enum indicating the out-of-bound status of the bounding box. It's statuses are described in
            :class:`OutOfBound`.
        conf_thre (float):
            A confidence score threshold that box is validated based on.
        conf_status (ConfidenceStatus):
            A :class:`ConfidenceStatus` enum indicating the status of bounding box's confidence.
        size_status (SizeStatus):
            A :class:`SizeStatus` enum indicating the status of bounding box's size.
        min_size_thre (Size):
            A minimum size threshold that box is validated based on.
    """

    def __init__(
        self,
        a_p1: Point2D,
        a_p2: Point2D,
        a_score: float,
        a_img_size: Optional[Size] = None,
        a_strict: Optional[bool] = False,
        a_conf_thre: Optional[float] = None,
        a_min_size_thre: Optional[Size] = None,
        a_do_validate: Optional[bool] = True,
        a_name: str = "BBox2D",
    ):
        """Constructor for BBox2D

        Args:
            a_p1 (Point2D):
                The top-left point of the bounding box.
            a_p2 (Point2D):
                The bottom-right point of the bounding box.
            a_score (float):
                The confidence score associated with the bounding box.
            a_name (str, optional):
                A string specifying the name of the bounding box (default is 'BBox2D').
            a_img_size (Size, optional):
                The :class:`Size` of the image the bounding box is associated with (default is 'None').
            a_strict (bool, optional):
                A flag indicating whether to enforce strict conditions for the bounding box (default is False).
            a_conf_thre (float, optional):
                A confidence score threshold that box can be validated based on (default is 'None').
            a_min_size_thre (Size, optional):
                A minimum :class:`Size` of a valid bounding box (default is 'None').
            a_do_validate (bool, optional):
                A bool flag to indicate the bounding box must be validated or not (default is `True`).
        """
        self.do_validate: bool = a_do_validate
        self.img_size: Optional[Size] = a_img_size
        self.strict: bool = a_strict
        self.conf_thre: Optional[bool] = a_conf_thre
        self.min_size_thre: Optional[Size] = a_min_size_thre
        # Statuses
        self.coord_status: CoordStatus = CoordStatus.UNKNOWN
        self.conf_status: ConfidenceStatus = ConfidenceStatus.UNKNOWN
        self.size_status: SizeStatus = SizeStatus.UNKNOWN

        # Validate inputs
        if self.do_validate:
            p1, p2, score = self._validate(a_p1=a_p1, a_p2=a_p2, a_score=a_score)
        else:
            p1, p2, score = a_p1, a_p2, a_score

        # Initialize bbox
        super().__init__(a_p1=p1, a_p2=p2, a_name=a_name)
        self.score = score

    def to_dict(self) -> dict:
        """Convert BBox2D to a dictionary

        Returns:
            dict: A dictionary representing the :class:`BBox2D` object.
        """
        dic = {"p1": self.p1.to_dict(), "p2": self.p2.to_dict(), "score": self.score}
        return dic

    @property
    def do_validate(self) -> bool:
        return self._do_validate

    @do_validate.setter
    def do_validate(self, a_do_validate: bool):
        if a_do_validate is not None and not isinstance(a_do_validate, bool):
            raise TypeError("The `a_do_validate` should be a `bool`.")
        self._do_validate: bool = a_do_validate

    @property
    def score(self) -> float:
        """Getter for the confidence score property.

        Returns:
            float: The confidence score [0, 1] associated with the bounding box.
        """
        return self._score

    @score.setter
    def score(self, a_score: float):
        """Setter for the confidence score property.

        Note:
            If the score is not in the range of [0, 1], then it will be scaled.

        Args:
            a_score (float):
                The confidence score to set.

        Raises:
            TypeError: If the input score is not a float.
        """
        float_flag = is_float(a_score)
        if a_score is None and not float_flag:
            raise TypeError("The `a_score` should be a float.")
        if float_flag:
            a_score = float(a_score)
            if a_score > 1.0:
                a_score = a_score / 100.0
        self._score: float = a_score

    @property
    def img_size(self) -> Size:
        """Get the :class:`Size` of the image associated with the bounding box.

        Returns:
            Size: The size of the image.
        """
        return self._img_size

    @img_size.setter
    def img_size(self, a_img_size: Size = None):
        """Set the :class:`Size` of the image associated with the bounding box.

        Args:
            a_img_size (Size, optional):
                The size of the image (default is `None`).

        Raises:
            TypeError: If the input type is not as expected.
        """
        if a_img_size is not None and not isinstance(a_img_size, Size):
            raise TypeError("The `a_image_size` should be a `Size`.")
        self._img_size: Size = a_img_size

    @property
    def conf_thre(self) -> float:
        """Get the confidence threshold

        Returns:
            float: The confidence threshold as a float.
        """
        return self._conf_thre

    @conf_thre.setter
    def conf_thre(self, a_conf_thre: float):
        """Set the confidence threshold

        Args:
            a_conf_thre (float, optional):
                A confidence score threshold that box can be validated based on (default is 'None').

        Raises:
            TypeError: If the input type is not as expected.
        """
        if a_conf_thre is not None and not isinstance(a_conf_thre, float):
            raise TypeError("The `a_conf_thre` should be a `float`.")
        self._conf_thre: float = a_conf_thre

    @property
    def min_size_thre(self) -> Size:
        """Get the :class:`Size` as the minimum size threshold of accepting a valid bounding box.

        Returns:
            Size: The minimum size threshold.
        """
        return self._min_size_thre

    @min_size_thre.setter
    def min_size_thre(self, a_min_size_thre: Size = None):
        """Set the minimum :class:`Size` threshold of the bounding box.

        Args:
            a_min_size_thre (Size, optional):
                The minimum size threshold of accepting a bounding box (default is 'None').

        Raises:
            TypeError: If the input type is not as expected.
        """
        if a_min_size_thre is not None and not isinstance(a_min_size_thre, Size):
            raise TypeError("The `a_min_size_thre` should be a `Size`.")
        self._min_size_thre: Size = a_min_size_thre

    @property
    def strict(self) -> bool:
        """Get the strictness condition for bounding box rules.

        Returns:
            bool: True if strict rules are applied, False otherwise.
        """
        return self._strict

    @strict.setter
    def strict(self, a_strict: bool):
        """Set the strictness condition for bounding box rules.

        Args:
            a_strict (bool):
                True if strict rules should be applied, False otherwise.

        Raises:
            TypeError: If the input type is not as expected.
        """
        if a_strict is None or not isinstance(a_strict, bool):
            raise TypeError("The `a_strict` should be a `bool`.")
        self._strict: bool = a_strict

    def to_xyxys(self) -> np.ndarray:
        """Convert the bounding box coordinates to a NumPy array [x, y, x, y, s].

        Returns:
            np.ndarray: A NumPy array containing the coordinates of both corner points and the confidence score.
        """
        return np.concatenate((self.p1.to_numpy(), self.p2.to_numpy(), [self.score]))

    def _validate_dtypes(
        self, a_p1: Point2D, a_p2: Point2D, a_score: float
    ) -> Tuple[Point2D, Point2D, float]:
        """Validate and convert the data types of the bounding box attributes.

        This method ensures that the data types of the bounding box attributes (points and score) are correct and
        consistent.

        Args:
            a_p1 (Point2D): The top-left point of the bounding box.
            a_p2 (Point2D): The bottom-right point of the bounding box.
            a_score (float): The confidence score associated with the bounding box.

        Returns:
            Tuple[Point2D, Point2D, float]: A tuple containing the validated and converted points and score.
        """
        # Convert the data types of points to be int
        a_p1.to_int()
        a_p2.to_int()

        # Convert score data type to be a float
        a_score = float(a_score)

        return a_p1, a_p2, a_score

    def _validate_xyxy(self, a_p1: Point2D, a_p2: Point2D) -> None:
        """Validate xyxy coordinates of the bounding box.

        This method checks the format and validity of the xyxy coordinates of the bounding box, updating the
        out-of-bound status accordingly.

        Args:
            a_p1 (Point2D): The top-left point of the bounding box.
            a_p2 (Point2D): The bottom-right point of the bounding box.

        Raises:
            ValueError: If the bounding box format is incorrect or if points are out of bounds in strict mode.
        """
        if a_p1.x > a_p2.x or a_p1.y > a_p2.y:
            if self.strict:
                raise ValueError(
                    "Incorrect BoundingBox2D format. It must be in type [x, y, x, y]."
                )
            self.coord_status = CoordStatus.INVALID_COORDINATES
        elif a_p1 == a_p2:
            if self.strict:
                raise ValueError(
                    "Given top-left and bottom-right points must be distinct."
                )
            self.coord_status = CoordStatus.INVALID_COORDINATES
        elif (self.img_size is not None) and (
            (a_p2.x <= 0 and a_p2.y <= 0)
            or (a_p1.x >= self.img_size.width and a_p1.y >= self.img_size.height)
        ):
            if self.strict:
                raise ValueError(
                    "Given bounding box's points are entirely out of bounds."
                )
            self.coord_status = CoordStatus.ENTIRELY_OOB
        elif (self.img_size is not None) and (
            a_p1.x < 0
            or a_p1.y < 0
            or a_p2.x > self.img_size.width
            or a_p2.y > self.img_size.height
        ):
            if self.strict:
                raise ValueError(
                    "Given bounding box's points are partially out of bounds."
                )
            self.coord_status = CoordStatus.PARTIALLY_OOB
        elif self.img_size is not None:
            self.coord_status = CoordStatus.VALID

    def _validate_score(self, a_score: float):
        """Validate the confidence score.

        This method validates the confidence score, normalizing it if necessary, and updates the confidence status
        accordingly.

        Args:
            a_score (float): The confidence score associated with the bounding box.

        Raises:
            ValueError: If the confidence score is below the specified threshold in strict mode.
        """
        if a_score > 1.0:
            a_score = a_score / 100.0
        if self.conf_thre is not None and a_score < self.conf_thre:
            if self.strict:
                raise ValueError("Given bounding box is not confident")
            self.conf_status = ConfidenceStatus.NOT_CONFIDENT
        elif self.conf_thre is not None:
            self.conf_status = ConfidenceStatus.CONFIDENT

    def _validate_size(self, a_size: Size):
        """
        Validates a bounding box based on its size against a minimum size threshold.

        This method compares the provided Size object with a minimum size threshold, updating the internal
        `size_status` attribute accordingly.

        Args:
            a_size (Size): The Size object representing the dimensions of the bounding box.

        Raises:
            ValueError: If the provided bounding box size is smaller than the specified threshold
                        and the validation is strict.

        Note:
            The `size_status` attribute is updated based on the validation result:
            - If the bounding box size is smaller than the threshold, and the validation is strict,
              a `ValueError` is raised. Otherwise, `size_status` is set to SizeStatus.INVALID.
            - If the bounding box size meets or exceeds the threshold, `size_status` is set to SizeStatus.VALID.
        """
        if (self.min_size_thre is not None) and not (a_size >= self.min_size_thre):
            if self.strict:
                raise ValueError("Given bounding box is smaller than the threshold")
            self.size_status = SizeStatus.INVALID
        elif self.min_size_thre is not None:
            self.size_status = SizeStatus.VALID

    def _validate(
        self,
        a_p1: Point2D,
        a_p2: Point2D,
        a_score: float,
    ) -> Tuple[Point2D, Point2D, float]:
        """Validate input parameters for BBox2D.

        This method performs validation on the input parameters to ensure correct data types, valid xyxy coordinates,
        valid size, and a valid confidence score.

        Args:
            a_p1 (Point2D): The top-left point of the bounding box.
            a_p2 (Point2D): The bottom-right point of the bounding box.
            a_score (float): The confidence score associated with the bounding box.

        Returns:
            Tuple[Point2D, Point2D, float]: A tuple containing validated top-left point, bottom-right point, and
            confidence score.
        """
        # Correct data types
        p1, p2, score = self._validate_dtypes(a_p1, a_p2, a_score)

        # Validate xyxy coordinates
        self._validate_xyxy(a_p1=p1, a_p2=p2)

        # Validate size
        size = Size(a_width=int(p2.x - p1.x), a_height=int(p2.y - p1.y))
        self._validate_size(a_size=size)

        # Validate confidence score
        self._validate_score(a_score=score)
        return p1, p2, score

    @classmethod
    def validate_array(cls, a_coordinates: Union[Tuple, List, np.ndarray]) -> None:
        """Validate coordinates array

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in [x, y, x, y, s] format.

        Returns:
            This method does not return any values
        Raises:
            TypeError: If the input coordinates are not of type Tuple, List, or np.ndarray.
            ValueError: If the input coordinates do not have a length of 5 or have more than one dimension.
        """
        if a_coordinates is None and not isinstance(
            a_coordinates, (Tuple, List, np.ndarray)
        ):
            raise TypeError(
                "The `a_coordinates` should be a `Tuple, List, or np.ndarray`."
            )

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 5:
            raise ValueError(
                f"`a_coordinates` array should have at least length 5 but it is in shape of"
                f" {a_coordinates.shape}."
            )

        if a_coordinates.ndim > 1:
            raise ValueError(
                f"`a_coordinates` array should be a 1D array but it has {a_coordinates.ndim} dimensions."
            )

    @classmethod
    def from_xyxys(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2D":
        """Create a BBox2D instance from coordinates in [x, y, x, y, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in [x, y, x, y, s] format.

            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2D: An instance of the BBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            **kwargs,
        )
        return box

    @classmethod
    def from_xywhs(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2D":
        """Create a BBox2D instance from coordinates in [x, y, w, h, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in [x, y, w, h, s] format.

            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2D: An instance of the BBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates)

        # Reformat xywh to xyxy
        a_coordinates[:4] = xywh_to_xyxy(*a_coordinates[:4])

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            **kwargs,
        )
        return box

    @classmethod
    def from_cxyars(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2D":
        """Create a BBox2D instance from coordinates in (center-x, center-y, area, aspect_ratio, score) format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in (center-x, center-y, area, aspect_ratio, score) format.

            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2D: An instance of the BBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates)

        # Reformat (center-x, center-y, area, aspect_ratio) to (x, y, x, y)
        a_coordinates[:4] = cxyar_to_xyxy(*a_coordinates[:4])

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            **kwargs,
        )
        return box

    @classmethod
    def from_cxywhs(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2D":
        """Creates a BBox2D instance from coordinates in (center-x, center-y, width, height, score) format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in (center-x, center-y, width, height, score) format.

            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2D: An instance of the BBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates)

        # Reformat (center-x, center-y, width, height) to (x, y, x, y)
        a_coordinates[:4] = cxywh_to_xyxy(*a_coordinates[:4])

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            **kwargs,
        )
        return box

    def __iter__(self):
        """Iterator for the BBox2D object.

        Yields:
            Union[int, float]: The xyxy values in sequence.
        """
        yield self.p1.x
        yield self.p1.y
        yield self.p2.x
        yield self.p2.y
        yield self.score

    def __getitem__(self, index):
        """Get an element from the BBox2D object by index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            Union[int, float]: The value at the specified index.

        Raises:
            IndexError: If the index is out of range for the BBox2D object.
        """
        if index == 0:
            return self.p1.x
        elif index == 1:
            return self.p1.y
        elif index == 2:
            return self.p2.x
        elif index == 3:
            return self.p2.y
        elif index == 4:
            return self.score
        else:
            raise IndexError("Index out of range for BBox2D object")

    def clamp(self) -> None:
        """Clamp the bounding box coordinates within the image boundaries.

        This method clamps the top-left and bottom-right coordinates of the bounding box to ensure that they fall
        within the boundaries of the associated image. If the bounding box is not out of bounds (PARTIALLY_OOB),
        and the image size is provided, the coordinates are adjusted accordingly.

        Note:
            This method modifies the bounding box in-place.

        Raises:
            ValueError: If the clamped coordinates do not meet the bounding box requirements.
        """
        if (
            self.coord_status == CoordStatus.PARTIALLY_OOB
            or self.coord_status == CoordStatus.VALID
            or self.coord_status == CoordStatus.UNKNOWN
        ) and self.img_size is not None:
            p1 = Point2D(a_x=max(int(self.p1.x), 0), a_y=max(int(self.p1.y), 0))
            p2 = Point2D(
                a_x=min(int(self.p2.x), self.img_size.width),
                a_y=min(int(self.p2.y), self.img_size.height),
            )

            # Validate the clamped coordinates
            self._validate_xyxy(a_p1=p1, a_p2=p2)

            # Set the clamped coordinates
            self.p1 = p1
            self.p2 = p2

    def scale(self, a_factor: float):
        """Scale the bounding box by a given factor.

        Args:
            a_factor (float): The scaling factor.

        Raises:
            TypeError: If the input factor is not a float.
            ValueError: If the factor is not a positive value.

        """
        float_flag = is_float(a_factor)
        if a_factor is None and not float_flag:
            raise TypeError("The `a_score` should be a float.")

        if a_factor <= 0:
            raise ValueError("The `a_factor` must be a positive value.")

        width = self.width * np.sqrt(a_factor)
        height = self.height * np.sqrt(a_factor)
        p1 = Point2D(a_x=self.center.x - width / 2, a_y=self.center.y - height / 2)
        p2 = Point2D(a_x=self.center.x + width / 2, a_y=self.center.y + height / 2)

        # Validate xyxy coordinates
        self._validate_xyxy(a_p1=p1, a_p2=p2)

        # Validate size
        size = Size(a_width=int(p2.x - p1.x), a_height=int(p2.y - p1.y))
        self._validate_size(a_size=size)

        self.p1 = p1
        self.p2 = p2

    def shift(self, a_size: Size):
        """Shift the bounding box by a given size.

        Args:
            a_size (:class:`Size`): The size by which to shift the bounding box.

        Raises:
            TypeError: If the input size is not an instance of :class:`Size`.

        """
        if a_size is None and not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")

        p1 = Point2D(a_x=self.p1.x + a_size.width, a_y=self.p1.y + a_size.height)
        p2 = Point2D(a_x=self.p2.x + a_size.width, a_y=self.p2.y + a_size.height)

        # Validate xyxy coordinates
        self._validate_xyxy(a_p1=p1, a_p2=p2)

        # Validate size
        size = Size(a_width=int(p2.x - p1.x), a_height=int(p2.y - p1.y))
        self._validate_size(a_size=size)

        self.p1 = p1
        self.p2 = p2


class BBox2DList(Box2DList, BaseObjectList[BBox2D]):
    """BBox2D List

    The BBox2DList class is based on the :class:`ObjectList` class and serves as a container for a collection of
    :class:`BBox2D` objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the BBox2DList (default is 'BBox2DList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[BBox2D], optional):
            A list of BBox2D objects to initialize the BBox2DList (default is None).
    """

    def __init__(
        self,
        a_name: str = "BBox2DList",
        a_max_size: int = -1,
        a_items: List[BBox2D] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    @classmethod
    def validate_array(
        cls, a_coordinates: Union[Tuple, List, np.ndarray]
    ) -> np.ndarray:
        """Validate coordinates array

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in [x, y, x, y, s] format.

        Returns:
            np.ndarray: The validated coordinates as an :class:`np.ndarray`.

        Raises:
            TypeError: If the input coordinates are not of type Tuple, List, or np.ndarray.
            ValueError: If the input coordinates do not have a length of 5 in the last dimension or have more than one
            dimension.
        """
        if a_coordinates is None and not isinstance(
            a_coordinates, (Tuple, List, np.ndarray)
        ):
            raise TypeError(
                "The `a_coordinates` should be a `Tuple, List, or np.ndarray`."
            )

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 5:
            raise ValueError(
                f"`a_coordinates` array should have length 5 in the last dimension(-1) but it has "
                f"{a_coordinates.shape}."
            )

        if a_coordinates.ndim == 1:
            a_coordinates = a_coordinates[np.newaxis]

        return a_coordinates

    @classmethod
    def from_xyxys(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2DList":
        """Create a BBox2DList instance from coordinates in [x, y, x, y, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the boxes in [x, y, x, y, s] format.
            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2DList: An instance of the BBox2DList class.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate bounding boxes
        bboxes = BBox2DList()
        bboxes.append(
            a_item=[BBox2D.from_xyxys(coord, **kwargs) for coord in coordinates]
        )
        return bboxes

    @classmethod
    def from_xywhs(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2DList":
        """Create a BBox2DList instance from coordinates in [x, y, w, h, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the boxes in [x, y, w, h, s] format.
            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2DList: An instance of the BBox2DList class.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate bounding boxes
        bboxes = BBox2DList()
        bboxes.append(
            a_item=[BBox2D.from_xywhs(coord, **kwargs) for coord in coordinates]
        )
        return bboxes

    @classmethod
    def from_cxywhs(
        cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs
    ) -> "BBox2DList":
        """Create a BBox2DList instance from coordinates in [center-x, center-y, width, height, score] format.

        This class method takes coordinates in the [center-x, center-y, width, height, score] format and creates a
        BBox2DList instance. It validates the input array, instantiates a BBox2DList, and populates it with BBox2D
        objects created from the provided coordinates using the from_cxywhs method of the BBox2D class.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding boxes in [center-x, center-y, width, height, score] format.
            **kwargs:
                Additional keyword arguments to pass to the BBox2D constructor.

        Returns:
            BBox2DList: An instance of the BBox2DList class.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate bounding boxes
        bboxes = BBox2DList()
        bboxes.append(
            a_item=[BBox2D.from_cxywhs(coord, **kwargs) for coord in coordinates]
        )
        return bboxes

    def clamp(self) -> None:
        """Clamp the coordinates of all bounding boxes

        This method iterates over all bounding boxes in the list and clamps their top-left and bottom-right coordinates
        within the boundaries of the associated image. If a bounding box is not out of bounds (PARTIALLY_OOB),
        and the image size is provided, the coordinates are adjusted accordingly.

        Note:
            This method modifies the bounding boxes in-place.
        """
        for box in self.items:
            box.clamp()

    def scale(self, a_factor: float) -> None:
        """Scale all bounding boxes in the list by a given factor.

        Args:
            a_factor (float): The scaling factor.

        """
        for box in self.items:
            box.scale(a_factor=a_factor)

    def shift(self, a_size: Size) -> None:
        """Shift all bounding boxes in the list by a given size.

        Args:
            a_size (:class:`Size`): The size by which to shift the bounding boxes.

        """
        for box in self.items:
            box.shift(a_size=a_size)

    def remove(
        self,
        a_status: Union[
            CoordStatus,
            ConfidenceStatus,
            SizeStatus,
            List[Union[CoordStatus, ConfidenceStatus, SizeStatus]],
        ],
    ):
        """
        Removes items from the list based on specified status or statuses.

        Args:
            a_status (Union[CoordStatus, ConfidenceStatus, SizeStatus, List[Union[CoordStatus, ConfidenceStatus,
             SizeStatus]]]):
                A single status or a list of statuses to identify items for removal.

        Raises:
            ValueError: If the `a_status` argument is None.

            TypeError:
                If any element in the `a_status` list is not of type `CoordStatus`, `ConfidenceStatus`, or
                `SizeStatus`.

        Returns:
            None: The method modifies the list in place.
        """
        if a_status is None:
            raise ValueError("The `a_status` argument cannot be None.")

        # Convert single status to a list
        if not isinstance(a_status, list):
            a_status = [a_status]

        if not all(
            isinstance(status, (CoordStatus, ConfidenceStatus, SizeStatus))
            for status in a_status
        ):
            raise TypeError(
                "All elements in the `a_status` list should be either `CoordStatus` or "
                "`ConfidenceStatus`, `SizeStatus`."
            )

        for index in reversed(range(len(self.items))):
            item = self.items[index]
            if any(
                (isinstance(status, CoordStatus) and item.coord_status == status)
                or (isinstance(status, ConfidenceStatus) and item.conf_status == status)
                or (isinstance(status, SizeStatus) and item.size_status == status)
                for status in a_status
            ):
                del self.items[index]

    def to_xyxys(self) -> np.ndarray:
        """Convert the BBox2DList to a NumPy array [[x, y, x, y, s], ...].

        Returns:
            np.ndarray:
                A 2D NumPy array representing the concatenation of the `to_numpy` result for each Box2D object in the
                list.
        """
        if len(self.items):
            arr = np.vstack([box.to_xyxys() for box in self.items])
        else:
            arr = np.empty(shape=(0, 5))

        return arr


class BBox2DNestedList(BaseObjectList[BBox2DList]):
    """Batch of 2D Bounding Boxes Lists

    This class extends the BaseObjectList and serves as a container for a collection of BBox2DList instances,
    representing grouped boxes. It provides methods for managing and manipulating the list of grouped boxes.

    Attributes:
        Inherits attributes from :class:`BaseObjectList`.
    """

    def __init__(
        self,
        a_name: str = "BBox2DNestedList",
        a_max_size: int = -1,
        a_items: List[BBox2DList] = None,
    ):
        """Initialize a BBox2DNestedList instance.

        Args:
            a_name (str, optional): The name of the BBox2DNestedList. Defaults to "BBox2DNestedList".
            a_max_size (int, optional): The maximum size of the list. Defaults to -1, indicating no size limit.
            a_items (List[BBox2DList], optional):
                A list of BBox2DList objects to initialize the BBox2DNestedList. Defaults to None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    def to_xyxys(self) -> np.ndarray:
        """Convert the BBox2DNestedList to a NumPy array [[x, y, x, y, s], ...].

        This method converts the grouped bounding boxes lists to a numpy array of bounding boxes in xyxys format,
        where each row represents a bounding box in the format [x1, y1, x2, y2, score].

        Returns:
            np.ndarray:
                A numpy array containing bounding boxes in xyxys format.
        """
        if len(self.items):
            arr = np.vstack([box_list.to_xyxys() for box_list in self.items])
        else:
            arr = np.empty(shape=(0, 5))
        return arr
