import numpy as np
import scipy as sp


def m_per_sec(km_per_h):
    return km_per_h * 1000 / (60 * 60)


def rad_per_sec(deg_per_sec):
    return np.radians(deg_per_sec)


def normalize_vector(v):
    return v / np.linalg.norm(v)


def lognormexp(values):
    return values - sp.special.logsumexp(values)
