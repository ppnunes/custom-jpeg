
import cv2
import numpy as np


def customDCT(block):

    imf = np.float32(block) / 255.0  # float conversion/scale
    dst = cv2.dct(imf)           # the dct
    img = np.uint8(dst) * 255.0
