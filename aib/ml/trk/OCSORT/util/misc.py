"""OC-SORT Miscellaneous Utilities

    This file defines the miscellaneous utilities of the OC-SORT.
"""

# region Imported Dependencies
from aib.ml.trk.OCSORT import StateDict

# endregion Imported Dependencies


def k_previous_obs(observations: StateDict, cur_age: int, k: int):
    """
    Selects the K-th previous observation states based on the age and delta_time(k) factor.

    Args:
        observations (StateDict): Dictionary containing observation states.
        cur_age (int): Current age of the tracker.
        k (int): The desired time delay.

    Returns:
        list: A list containing the selected kth previous observation states in the format [x1, y1, x2, y2, score].

    Note:
        The function extracts the kth previous observation based on the age and delta_time factor.
        If the exact age is not available in the observations, the observation with the maximum age is selected.
    """
    result = [-1, -1, -1, -1, -1]
    if len(observations) > 0:
        for i in range(k):
            dt = k - i
            if cur_age - dt in observations:
                result = observations[cur_age - dt].box.to_xyxys()
                break
        else:
            max_age = max(observations.keys())
            result = observations[max_age].box.to_xyxys()
    return result
