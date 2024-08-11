"""Association Result
"""

# region Imported Dependencies
from brain.util.ml.reid.util.assoc import MTEList, UMTList, UMEList
from brain.util.obj import BaseObject

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class Associations(BaseObject):
    def __init__(self, a_name: str = "Associations"):
        super().__init__(a_name=a_name)
        self.matched_pairs: MTEList = MTEList()
        self.unmatched_targets: UMTList = UMTList()
        self.unmatched_entities: UMEList = UMEList()

    def to_dict(self) -> dict:
        dic = {
            "Matched Pairs": self.matched_pairs,
            "Unmatched Targets": self.unmatched_targets,
            "Unmatched Entities": self.unmatched_entities,
        }
        return dic
