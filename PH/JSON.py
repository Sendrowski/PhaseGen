import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Convert unserializable numpy arrays to lists.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save(d: dict, filepath):
    with open(filepath, 'w') as fh:
        json.dump(d, fh, indent=4, cls=NumpyEncoder)
