"""Small dependency-free subset of the PyBullet quaternion helpers.

AMP data loading only needs quaternion multiplication, inverse, axis-angle,
and SLERP.  Keeping these helpers local makes the training framework usable
without requiring the PyBullet package just to import an AMP runner.
"""

import numpy as np


def quaternion_normalize(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def quaternion_inverse(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    result = quaternion.copy()
    result[..., :3] *= -1.0
    return result / np.sum(np.square(quaternion), axis=-1, keepdims=True)


def quaternion_multiply(q0, q1):
    x0, y0, z0, w0 = np.moveaxis(np.asarray(q0, dtype=np.float64), -1, 0)
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1, dtype=np.float64), -1, 0)
    return np.stack(
        (
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ),
        axis=-1,
    )


def quaternion_about_axis(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = 0.5 * angle
    return np.concatenate((axis * np.sin(half_angle), [np.cos(half_angle)]))


def quaternion_slerp(q0, q1, fraction):
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    fraction = np.asarray(fraction)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    linear = sin_theta < 1e-7
    a = np.sin((1.0 - fraction) * theta) / np.maximum(sin_theta, 1e-7)
    b = np.sin(fraction * theta) / np.maximum(sin_theta, 1e-7)
    result = a * q0 + b * q1
    result = np.where(linear, (1.0 - fraction) * q0 + fraction * q1, result)
    return quaternion_normalize(result)
