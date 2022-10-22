import numpy as np
import cv2
import math
from transformations import *
import os
from os import listdir
from os.path import isfile, join
from locate import locate
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    frame = cv2.rotate(img, cv2.ROTATE_180)
    detected = find(frame, save_result=False)
    cv2.imshow('detected',detected)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break   
cap.release()
cv2.destroyAllWindows()
print("end")