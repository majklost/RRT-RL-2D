import numpy as np


def rot_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi
