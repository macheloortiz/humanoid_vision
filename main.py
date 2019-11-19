import numpy as np
import cv2
import math
from transformations import *
import os
from os import listdir
from os.path import isfile, join

def locate(file_name, save_result=False):
    img = cv2.imread(file_name)

    camera_height = 40
    camera_target_distance = 120
    camera_angle = math.pi - math.atan2(camera_target_distance, camera_height)
    fov = np.deg2rad(60/2)
    # width = 640
    # height = 480

    width = 4160
    height = 3120

    f = width/2 / math.tan(fov)

    a = 10
    ring_radious = 75

    p1 = np.array([-a, camera_target_distance+a, 0, 1])
    p2 = np.array([a, camera_target_distance+a, 0, 1])
    p3 = np.array([a, camera_target_distance-a, 0, 1])
    p4 = np.array([-a, camera_target_distance-a, 0, 1])

    camera_transformation = translation(np.array([0, 0, camera_height, 0])).dot(rotation_x(-camera_angle))

    inv_cam_trans = np.linalg.inv(camera_transformation)

    p1_c = inv_cam_trans.dot(p1)
    p2_c = inv_cam_trans.dot(p2)
    p3_c = inv_cam_trans.dot(p3)
    p4_c = inv_cam_trans.dot(p4)

    def get_pixel_coords(f, p, width, height):
        return np.array([-f/p[2]*p[0]+width/2, f/p[2]*p[1]+height/2, 0, 1])

    p1_i = get_pixel_coords(f, p1_c, width, height)
    p2_i = get_pixel_coords(f, p2_c, width, height)
    p3_i = get_pixel_coords(f, p3_c, width, height)
    p4_i = get_pixel_coords(f, p4_c, width, height)

    img2 = cv2.circle(img, (int(p1_i[0]), int(p1_i[1])), 10, (0,0,0))
    img2 = cv2.circle(img2, (int(p2_i[0]), int(p2_i[1])), 10, (100,100,100))
    img2 = cv2.circle(img2, (int(p3_i[0]), int(p3_i[1])), 10, (0,0,0))
    img2 = cv2.circle(img2, (int(p4_i[0]), int(p4_i[1])), 10, (255,255,255))

    src_pts = np.array([[p1_i[0], p1_i[1]],
                  [p2_i[0], p2_i[1]],
                  [p3_i[0], p3_i[1]],
                  [p4_i[0], p4_i[1]]], dtype="float32")

    px_per_cm = 1
    rect_res = a*px_per_cm
    center_width = int(ring_radious*1.5*px_per_cm)
    center_height = int(ring_radious*1.5*px_per_cm)
    dst_pts = np.array([[center_width + rect_res, center_height - rect_res],
                  [center_width - rect_res, center_height - rect_res],
                  [center_width - rect_res, center_height + rect_res],
                  [center_width + rect_res, center_height + rect_res]], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(src_pts,dst_pts)
    warped = cv2.warpPerspective(img, perspective_matrix, (3*ring_radious*px_per_cm, 3*ring_radious*px_per_cm))
    processed = cv2.GaussianBlur(warped, (int(ring_radious/10), int(ring_radious/10)), cv2.BORDER_DEFAULT)
    processed = cv2.cvtColor(processed,cv2.COLOR_BGR2GRAY)

    # cv2.imshow('processed', processed)
    # cv2.imshow('nic2', warped)
    edges = cv2.Canny(processed, 20, 200)
    # cv2.imshow('canny', edges)

    def find_best_circle(img, radious):
        max_threshold = 100
        min_threshold = -92
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1,20,
                                   param1=200, param2=int((max_threshold+min_threshold)/2),
                                   minRadius=int(radious*0.9),
                                   maxRadius=int(radious*1.1))
        while int(max_threshold)>int(min_threshold):
            if circles is None:
                max_threshold = (max_threshold + min_threshold)/2
            elif len(circles[0, :]) > 1:
                min_threshold = (max_threshold + min_threshold) / 2
            elif len(circles[0, :]) == 1:
                return circles[0,0]
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=200, param2=int((max_threshold+min_threshold)/2),
                                       minRadius=int(radious * 0.9),
                                       maxRadius=int(radious * 1.1))
        if len(circles[0, :]) == 0:
            return None
        return circles[0,0]


    circle = find_best_circle(processed, ring_radious*px_per_cm)
    circle = np.uint16(np.around(circle))
    # draw the outer circle
    cv2.circle(warped,(circle[0],circle[1]),circle[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(warped,(circle[0],circle[1]),1,(0,0,255),3)
    # cv2.imshow('detected circles',warped)

    if save_result:
        cv2.imwrite(f'detected_{file_name}', warped)


    sobel_x = np.array([[ -1, 0, 1],
                       [ -2, 0, 2],
                       [ -1, 0, 1]])
    sobel_y = np.array([[ -1, -2, -1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]])
    filtered_blurred_x = cv2.filter2D(processed, cv2.CV_32F, sobel_x)
    filtered_blurred_y = cv2.filter2D(processed, cv2.CV_32F, sobel_y)

    phase = cv2.phase(np.array(filtered_blurred_x, np.float32), np.array(filtered_blurred_y, dtype=np.float32), angleInDegrees=True)
    laplacian = cv2.Laplacian(processed,cv2.CV_64F)

    # cv2.imshow('phase', phase)
    # cv2.imshow('laplacian', laplacian)

    cv2.waitKey(0)
    print(camera_angle)


if __name__ == '__main__':
    for filename in listdir('images'):
        file = 'images/'+filename
        locate(file, True)