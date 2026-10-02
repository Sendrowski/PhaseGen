"""
Serialization mixin class.
"""

from typing import TYPE_CHECKING

import jsonpickle

if TYPE_CHECKING:
    from typing_extensions import Self


class Serializable:
    """
    Mixin class for serialization.
    """

    def to_file(self, file: str) -> None:
        """
        Save object to file (JSON).

        :param file: File path.
        """
        with open(file, 'w') as fh:
            fh.write(self.to_json())

    def to_json(self) -> str:
        """
        Serialize object.

        :return: JSON string
        """
        return jsonpickle.encode(self, indent=4, warn=True, keys=True)

    @classmethod
    def from_json(cls, json: str, classes=None) -> 'Self':
        """
        Restore an object from its JSON serialization.

        :param classes: Classes to be used for deserialization.
        :param json: JSON string
        :return: The restored object.
        """
        return jsonpickle.decode(json, classes=classes, keys=True)

    @classmethod
    def from_file(cls, file: str, classes=None) -> 'Self':
        """
        Restore an object from a file holding its JSON serialization.

        :param classes: Classes to be used for deserialization.
        :param file: File to load from.
        :return: The restored object.
        """
        with open(file, 'r') as fh:
            return cls.from_json(fh.read(), classes)
