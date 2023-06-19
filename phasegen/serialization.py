from typing import Self

import jsonpickle
import numpy as np
from jsonpickle.handlers import BaseHandler


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
        return jsonpickle.encode(self, indent=4, warn=True, make_refs=False)

    @classmethod
    def from_json(cls, json: str, classes=None) -> Self:
        """
        Unserialize object.

        :param classes: Classes to be used for unserialization
        :param json: JSON string
        """
        return jsonpickle.decode(json, classes=classes)

    @classmethod
    def from_file(cls, file: str, classes=None) -> Self:
        """
        Load object from file.

        :param classes: Classes to be used for unserialization
        :param file: File to load from
        """
        with open(file, 'r') as fh:
            return cls.from_json(fh.read(), classes)


class NumpyArrayHandler(BaseHandler):
    """
    Handler for numpy arrays.
    """

    def flatten(self, x: np.ndarray, data: dict) -> dict:
        """
        Convert Spectrum to dict.

        :param x: Numpy array
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=x.tolist())

    def restore(self, data: dict) -> np.ndarray:
        """
        Restore Spectrum.

        :param data: Dictionary
        :return: Numpy array
        """
        return np.array(data['data'])
