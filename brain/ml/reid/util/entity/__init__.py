"""Identification Entity Modules

This module provides subsequent modules of the Identification Entity modules.
"""

# region Imported Dependencies
from .state import ReidStateTable, ReidState, ReidStateList, TypeReidState
from .desc import ReidDesc, ReidDescList
from .ent import (
    ReidEntity,
    TypeReidEntity,
    ReidEntityList,
    ReidEntityDict,
    TypeReidEntityDict,
    ReidEntityNestedList,
)
from .tgt import (
    ReidTarget,
    TypeReidTarget,
    ReidTargetList,
    TypeReidTargetList,
    ReidTargetDict,
    TypeReidTargetDict,
    ReidTargetNestedList,
)

# endregion Imported Dependencies
