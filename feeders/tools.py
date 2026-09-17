import random

import numpy as np


def auto_pading(data_numpy, size, random_pad=False):
    c, t, v, m = data_numpy.shape
    if t >= size:
        return data_numpy[:, :size, :, :]
    begin = random.randint(0, size - t) if random_pad else 0
    out = np.zeros((c, size, v, m), dtype=data_numpy.dtype)
    out[:, begin:begin + t, :, :] = data_numpy
    return out


def random_choose(data_numpy, size, auto_pad=True):
    c, t, v, m = data_numpy.shape
    if t == size:
        return data_numpy
    if t < size:
        return auto_pading(data_numpy, size, random_pad=True) if auto_pad else data_numpy
    begin = random.randint(0, t - size)
    return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=(-10.0, -5.0, 0.0, 5.0, 10.0),
                scale_candidate=(0.9, 1.0, 1.1),
                transform_candidate=(-0.2, -0.1, 0.0, 0.1, 0.2)):
    c, t, v, m = data_numpy.shape
    if c < 2:
        return data_numpy
    angle = random.choice(angle_candidate) * np.pi / 180.0
    scale = random.choice(scale_candidate)
    tx = random.choice(transform_candidate)
    ty = random.choice(transform_candidate)
    rot = np.array(
        [[np.cos(angle) * scale, -np.sin(angle) * scale],
         [np.sin(angle) * scale, np.cos(angle) * scale]],
        dtype=data_numpy.dtype,
    )
    xy = data_numpy[:2].reshape(2, -1)
    xy = rot @ xy
    xy[0] += tx
    xy[1] += ty
    data_numpy[:2] = xy.reshape(2, t, v, m)
    return data_numpy


def random_shift(data_numpy):
    c, t, v, m = data_numpy.shape
    out = np.zeros_like(data_numpy)
    valid = (data_numpy != 0).sum(axis=(0, 2, 3)) > 0
    if not valid.any():
        return data_numpy
    begin = valid.argmax()
    end = len(valid) - valid[::-1].argmax()
    size = end - begin
    bias = random.randint(0, max(0, t - size))
    out[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]
    return out
