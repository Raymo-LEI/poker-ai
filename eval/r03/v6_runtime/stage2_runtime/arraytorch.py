import numpy as np
Tensor = np.ndarray
device = str
long = np.int64
float32 = np.float32
bool = np.bool_
def zeros(*shape, dtype=np.float32):
    return np.zeros(shape[0] if len(shape)==1 and isinstance(shape[0], tuple) else shape, dtype=dtype)
def ones(*shape, dtype=np.float32):
    return np.ones(shape[0] if len(shape)==1 and isinstance(shape[0], tuple) else shape, dtype=dtype)
def tensor(value, dtype=None):
    return np.array(value, dtype=dtype)
def from_numpy(value):
    return value
def cat(values, dim=0):
    return np.concatenate(values, axis=dim)
