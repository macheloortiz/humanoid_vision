import numpy as np
from math import sin, cos


def rotation_x(angle):
    return np.array([[1, 0, 0, 0],
                     [0, cos(angle), -sin(angle), 0],
                     [0, sin(angle), cos(angle), 0],
                     [0, 0, 0, 1]])

def rotation_y(angle):
    return np.array([[cos(angle), 0, sin(angle), 0],
                     [0, 1, 0, 0],
                     [-sin(angle), 0, cos(angle), 0],
                     [0, 0, 0, 1]])

def rotation_z(angle):
    return np.array([[cos(angle), -sin(angle), 0, 0],
                     [sin(angle), cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translation(vector):
    return np.array([[1,0,0,vector[0]],
                     [0,1,0,vector[1]],
                     [0,0,1,vector[2]],
                     [0,0,0,1]])
