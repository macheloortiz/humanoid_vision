import numpy as np
import cv2
import math
from transformations import *

img = cv2.imread('test.png')

camera_height = 20
camera_target_distance = 30
camera_angle = math.pi - math.atan2(camera_target_distance, camera_height)
fov = np.deg2rad(60 / 2)
width = 640
height = 480
f = width / 2 / math.tan(fov)

a = 10
ring_radious = 50

p1 = np.array([-a, camera_target_distance + a, 0, 1])
p2 = np.array([a, camera_target_distance + a, 0, 1])
p3 = np.array([a, camera_target_distance - a, 0, 1])
p4 = np.array([-a, camera_target_distance - a, 0, 1])

camera_transformation = translation(np.array([0, 0, camera_height, 0])).dot(rotation_x(-camera_angle))

inv_cam_trans = np.linalg.inv(camera_transformation)

p1_c = inv_cam_trans.dot(p1)
p2_c = inv_cam_trans.dot(p2)
p3_c = inv_cam_trans.dot(p3)
p4_c = inv_cam_trans.dot(p4)


def get_pixel_coords(f, p, width, height):
    return np.array([-f / p[2] * p[0] + width / 2, f / p[2] * p[1] + height / 2, 0, 1])


p1_i = get_pixel_coords(f, p1_c, width, height)
p2_i = get_pixel_coords(f, p2_c, width, height)
p3_i = get_pixel_coords(f, p3_c, width, height)
p4_i = get_pixel_coords(f, p4_c, width, height)

img2 = cv2.circle(img, (int(p1_i[0]), int(p1_i[1])), 10, (0, 0, 0))
img2 = cv2.circle(img2, (int(p2_i[0]), int(p2_i[1])), 10, (100, 100, 100))
img2 = cv2.circle(img2, (int(p3_i[0]), int(p3_i[1])), 10, (0, 0, 0))
img2 = cv2.circle(img2, (int(p4_i[0]), int(p4_i[1])), 10, (255, 255, 255))

src_pts = np.array([[p1_i[0], p1_i[1]],
                    [p2_i[0], p2_i[1]],
                    [p3_i[0], p3_i[1]],
                    [p4_i[0], p4_i[1]]], dtype="float32")

px_per_cm = 2
rect_res = a * px_per_cm
center_width = int(ring_radious * 1.5 * px_per_cm)
center_height = int(ring_radious * 1.5 * px_per_cm)
dst_pts = np.array([[center_width + rect_res, center_height - rect_res],
                    [center_width - rect_res, center_height - rect_res],
                    [center_width - rect_res, center_height + rect_res],
                    [center_width + rect_res, center_height + rect_res]], dtype="float32")

perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, perspective_matrix, (3 * ring_radious * px_per_cm, 3 * ring_radious * px_per_cm))

# mask = np.array(warped[:,:,0] == 0, dtype='uint8')
# mask = cv2.trheshold(mask, 0.5)
processed = cv2.medianBlur(warped, 5)
processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)


circles = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=15, minRadius=int(ring_radious * px_per_cm * 0.9),
                           maxRadius=int(ring_radious * px_per_cm * 1.1))
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(warped, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(warped, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('original', img)
ring = circles[0, 0]

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
filtered_blurred_x = cv2.filter2D(processed, cv2.CV_32F, sobel_x)
filtered_blurred_y = cv2.filter2D(processed, cv2.CV_32F, sobel_y)

phase = cv2.phase(np.array(filtered_blurred_x, np.float32), np.array(filtered_blurred_y, dtype=np.float32),
                  angleInDegrees=False)
laplacian = cv2.Laplacian(processed, cv2.CV_64F)

n_samples = 400
angle = 0

interruption = phase * 0
for sample in range(n_samples):
    # angle = -fov + 2*fov*(sample/n_samples) - math.pi/2
    ring_width = ring_radious * 0.1
    n_width_samples = 10
    angle = math.pi * 2 * (sample / n_samples)
    for width_sample in range(n_width_samples):
        radious_to_check = ring_radious - ring_width / 2 + ring_width * width_sample / n_width_samples
        px = np.array([ring[0] + radious_to_check * px_per_cm * math.cos(angle),
                       ring[1] + radious_to_check * px_per_cm * math.sin(angle)])
        if processed[int(px[1]), int(px[0])] == 0:
            continue
        processed[int(px[1]), int(px[0])] = 255
        interruption[int(px[1]), int(px[0])] = math.sin(phase[int(px[1]), int(px[0])] - angle) ** 2 * math.fabs(
            laplacian[int(px[1]), int(px[0])])

mask = cv2.inRange(processed, 1, 255)
kernel = np.ones((5,5), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
interruption = cv2.bitwise_or(interruption, interruption, mask=mask)

max_value = np.amax(interruption)
enemy_coord = np.where(interruption == np.amax(interruption))


cv2.imshow('mask', mask)
cv2.imshow('interruption', normalize(interruption))
cv2.imshow('processed', processed)
cv2.imshow('warped', warped)
cv2.imshow('phase', normalize(phase))
cv2.imshow('laplacian', normalize(laplacian))

cv2.waitKey(0)
print(camera_angle)
