import numpy as np
import cv2
import math
from transformations import *
# from locate import locate
#from matplotlib import pyplot as plt

def find_enemy(img, circle):
    radius = circle[2]
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    # processed = cv2.medianBlur(img, 5)
    # processed = cv2.GaussianBlur(img, (int(radius/10), int(radius/10)), cv2.BORDER_DEFAULT)
    processed = img
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    filtered_blurred_x = cv2.filter2D(processed, cv2.CV_32F, sobel_x)
    filtered_blurred_y = cv2.filter2D(processed, cv2.CV_32F, sobel_y)

    phase = cv2.phase(np.array(filtered_blurred_x, np.float32), np.array(filtered_blurred_y, dtype=np.float32),
                      angleInDegrees=False)
    laplacian = cv2.Laplacian(processed, cv2.CV_64F)

    n_samples = 400
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    interruption = phase * 0
    for sample in range(n_samples):
        # angle = -fov + 2*fov*(sample/n_samples) - math.pi/2
        ring_width = radius * 0.1
        n_width_samples = 5
        angle = math.pi * 2 * (sample / n_samples)
        for width_sample in range(n_width_samples):
            radius_to_check = radius - ring_width / 2 + ring_width * width_sample / n_width_samples
            px = np.array([circle[0] + radius_to_check * math.cos(angle),
                           circle[1] + radius_to_check * math.sin(angle)])
            if int(px[1])<0 or int(px[1])>=height or int(px[0])<0 or int(px[0])>=width or processed[int(px[1]), int(px[0])] == 0:
                continue
            processed[int(px[1]), int(px[0])] = 255
            interruption[int(px[1]), int(px[0])] = math.sin(phase[int(px[1]), int(px[0])] - angle) ** 2 * math.fabs(
                laplacian[int(px[1]), int(px[0])])

    mask = cv2.inRange(processed, 1, 255)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    interruption = cv2.bitwise_or(interruption, interruption, mask=mask)

    abs_enemy_det_thr = 100
    # No enemy found
    if interruption.max()<abs_enemy_det_thr:
        return 0, 0
    interruption = normalize(interruption)
    threshold = interruption.max() * 0.2

    _, thr_interruption = cv2.threshold(interruption, threshold, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thr_interruption)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 1, (0, 255, 0), 2)
    # enemy_coord = np.where(interruption == np.amax(interruption))
    # cv2.circle(img, (enemy_coord[1], enemy_coord[0]), 1, (0, 0, 255), 3)
    # plt.imshow(thr_interruption, 'gray')
    # plt.show()
    # cv2.imshow('interruption', normalize(interruption))
    # cv2.imshow('processed', processed)
    # cv2.imshow('warped', img)
    # cv2.imshow('phase', normalize(phase))
    # cv2.imshow('laplacian', normalize(laplacian))
    # cv2.waitKey(0)

    return cX, cY
