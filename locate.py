import numpy as np
import cv2
import math
from transformations import *
import os
from os import listdir
from os.path import isfile, join
from find_enemy import find_enemy


def get_pixel_coords(f, p, width, height):
    """
    This function calculates the pixel coordinates of a 3D space point given its position relative to the camera
    :param f: camera focal length (in pixels)
    :param p: point coordinates
    :param width: image width
    :param height: image height
    :return: 2D image coordinates in pixels
    """
    return np.array([-f / p[2] * p[0] + width / 2, f / p[2] * p[1] + height / 2, 0, 1])

def find_best_circle(img, radius):
    """
    A function that finds the most fitting circle in an image with a given size (+-10%)
    :param img: source image
    :param radius: radius of searched circle (in px)
    :return: best circle (x,y,r)
    """
    max_threshold = 100
    min_threshold = -92
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1,20,
                               param1=200, param2=int((max_threshold+min_threshold)/2),
                               minRadius=int(radius*0.9),
                               maxRadius=int(radius*1.1))
    while int(max_threshold)>int(min_threshold):
        if circles is None:
            max_threshold = (max_threshold + min_threshold)/2
        elif len(circles[0, :]) > 1:
            min_threshold = (max_threshold + min_threshold) / 2
        elif len(circles[0, :]) == 1:
            return circles[0,0]
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=200, param2=int((max_threshold+min_threshold)/2),
                                   minRadius=int(radius * 0.9),
                                   maxRadius=int(radius * 1.1))
    if len(circles[0, :]) == 0:
        return None
    return circles[0,0]


def locate(img, ring_radius=75, camera_height=40, camera_target_distance=120, fov_angle=60, save_result=False):
    """
    Function that detects the edge of dohyo and returns the distance to its center
    :param img: source image
    :param ring_radius: radius of the dohyo (in cm)
    :param camera_height: the height of camera above the dohyo
    :param camera_target_distance: the distance of camera target
    (the distance to the point on the dohyo that is in the center of the image), used to calculate the pitch of the camera
    :param fov_angle: horizontal field of view (in cm)
    :param save_result: Boolean for saving the image with detected circle overlaid
    :return:    -the distance to the center of the dohyo
                -the horizontal angle (0-towards the center, 90-center is 90 degrees to the right, 180-away from the center)
                -the angle to enemy (0-ahead, 90-center is 90 degrees to the right, 180-away from the center)
                -the warped image
                -the circle info
    """
    camera_angle = math.pi - math.atan2(camera_target_distance, camera_height)
    fov = np.deg2rad(fov_angle/2)
    # width = 640
    # height = 480

    width = np.shape(img)[1]
    height = np.shape(img)[0]

    # Calculating the focal length
    f = width/2 / math.tan(fov)

    # points p1-p4 are vertices of a virtual 2a x 2a square lying on the ground
    a = 10
    p1 = np.array([-a, camera_target_distance+a, 0, 1])
    p2 = np.array([a, camera_target_distance+a, 0, 1])
    p3 = np.array([a, camera_target_distance-a, 0, 1])
    p4 = np.array([-a, camera_target_distance-a, 0, 1])
    # Calculating the transformation matrix for the camera
    camera_transformation = translation(np.array([0, 0, camera_height, 0])).dot(rotation_x(-camera_angle))
    # Inverting the camera transformation matrix
    inv_cam_trans = np.linalg.inv(camera_transformation)
    # Calculating the position of the square vertices relative to the camera
    p1_c = inv_cam_trans.dot(p1)
    p2_c = inv_cam_trans.dot(p2)
    p3_c = inv_cam_trans.dot(p3)
    p4_c = inv_cam_trans.dot(p4)

    # Finding the position of the virtual square in the picture
    p1_i = get_pixel_coords(f, p1_c, width, height)
    p2_i = get_pixel_coords(f, p2_c, width, height)
    p3_i = get_pixel_coords(f, p3_c, width, height)
    p4_i = get_pixel_coords(f, p4_c, width, height)

    src_pts = np.array([[p1_i[0], p1_i[1]],
                  [p2_i[0], p2_i[1]],
                  [p3_i[0], p3_i[1]],
                  [p4_i[0], p4_i[1]]], dtype="float32")

    # Warpingthe perspective
    px_per_cm = 1
    rect_res = a*px_per_cm
    center_width = int(ring_radius*1.5*px_per_cm)
    center_height = int(ring_radius*1.5*px_per_cm)
    dst_pts = np.array([[center_width + rect_res, center_height - rect_res],
                  [center_width - rect_res, center_height - rect_res],
                  [center_width - rect_res, center_height + rect_res],
                  [center_width + rect_res, center_height + rect_res]], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(src_pts,dst_pts)
    warped = cv2.warpPerspective(img, perspective_matrix, (3*ring_radius*px_per_cm, 3*ring_radius*px_per_cm))

    # Image processing (blurring and converting to gray)
    processed = cv2.GaussianBlur(warped, (int(ring_radius/10), int(ring_radius/10)), cv2.BORDER_DEFAULT)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('processed', processed)
    # cv2.imshow('nic2', warped)
    # edges = cv2.Canny(processed, 20, 200)
    # cv2.imshow('canny', edges)

    circle = find_best_circle(processed, ring_radius*px_per_cm)
    circle = np.uint16(np.around(circle))

    detected = np.array(warped)
    cv2.circle(detected,(circle[0],circle[1]),circle[2],(0,255,0),2)
    cv2.circle(detected,(circle[0],circle[1]),1,(0,0,255),3)

    if save_result:
        cv2.imwrite(f'detected.png', detected)

    camera_position = np.array([center_width, center_height+camera_target_distance*px_per_cm])
    camera2center = np.array([circle[0], circle[1]]) - camera_position
    distance = np.linalg.norm(camera2center)
    angle = math.atan2(camera2center[0], -camera2center[1])

    x, y = find_enemy(warped, circle)
    enemy_angle = None
    if (x,y) != (0,0):
        enemy_pos = np.array([x, y])
        camera2enemy = enemy_pos-camera_position
        enemy_angle = math.atan2(camera2enemy[0], -camera2enemy[1])
    return distance, angle, enemy_angle, warped, circle


if __name__ == '__main__':
    img = cv2.imread('images/8.jpg')
    distance, angle, angle_enemy, warped, circle = locate(img)
    pass
