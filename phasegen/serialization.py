"""
Serialization mixin class.
"""

import jsonpickle


class Serializable:
    """
    Mixin class for serialization.
    """

    def to_file(self, file: str):
        """
        Save object to file.

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
        Unserialize object.

        :param classes: Classes to be used for deserialization.
        :param json: JSON string
        """
        return jsonpickle.decode(json, classes=classes, keys=True)

    @classmethod
    def from_file(cls, file: str, classes=None) -> 'Self':
        """
        Load object from file.

        :param classes: Classes to be used for unserialization
        :param file: File to load from
        """
        with open(file, 'r') as fh:
            return cls.from_json(fh.read(), classes)
