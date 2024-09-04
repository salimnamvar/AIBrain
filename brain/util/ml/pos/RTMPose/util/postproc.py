"""RTMPose Post-Processing Utilities
"""

# region Imported Dependencies
from typing import Tuple

import numpy as np
import numpy.typing as npt


# endregion Imported Dependencies


# TODO(doc): Complete the document of following function
def extract_keypoints(
    a_simcc_x: npt.NDArray[np.floating], a_simcc_y: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    # Reshape Input Data
    N, K, Wx = a_simcc_x.shape
    simcc_x = a_simcc_x.reshape(N * K, -1)
    simcc_y = a_simcc_y.reshape(N * K, -1)

    # Find Maximum Value Locations
    kps_x = np.argmax(simcc_x, axis=1)
    kps_y = np.argmax(simcc_y, axis=1)
    kps = np.stack((kps_x, kps_y), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # Combining the Confidence Scores
    scores = 0.5 * (max_val_x + max_val_y)

    # Invalid Detection Handling
    kps[scores <= 0.0] = -1

    # Reshape Key-points and Scores
    kps = kps.reshape(N, K, 2)
    scores = scores.reshape(N, K)

    return kps, scores
