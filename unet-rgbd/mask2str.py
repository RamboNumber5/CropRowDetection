# -*- coding:utf-8 -*-
# E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
# The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

import cv2
import numpy as np

# test = np.array([[0,1,0],[1,0,1]])
# print(np.where(test.flatten(order='F') == 1))
mask = cv2.imread('data/test/label/0.jpg', cv2.IMREAD_GRAYSCALE)

# Aloow printing the entire array without summarising with ellipses
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(mask)
# mask=255,background=0
bytes = np.where(mask.flatten(order='F') == 255)[0]
runs = []
prev = -2
for b in bytes:
    if (b > prev + 1):
        runs.extend((b + 1, 0))
    runs[-1] += 1
    prev = b

print(' '.join([str(i) for i in runs]))