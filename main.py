import numpy as np
import cv2
import math
from transformations import *
import os
from os import listdir
from os.path import isfile, join
from locate import locate

if __name__ == '__main__':
    for filename in listdir('images'):
        file = 'images/'+filename
        img = cv2.imread(file)
        locate(img, save_result=True)