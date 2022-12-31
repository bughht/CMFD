import os

import cv2 as cv
import numpy as np

import utils


class SIFT_Methods:
    def __init__(self):
        self.sift = cv.SIFT_create()


if __name__ == "__main__":
    df = utils.load_data_csv()
    img = cv.imread(os.path.join(utils.__rootdir__, df['img'][0]))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    img = cv.drawKeypoints(
        img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
