"""Association Result
"""

# region Imported Dependencies
from aib.legacy.reid.util.assoc import MTEList, UMTList, UMEList
from aib.obj import ExtBaseObject

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class Associations(ExtBaseObject):
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
