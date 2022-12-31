import os

import cv2 as cv
import numpy as np

import utils


class SIFT_Methods:
    def __init__(self, Preprocessing):
        self.sift = cv.SIFT_create()
        self.Preprocessing = Preprocessing

    def feature_extraction(self, img):
        img_pre = self.Preprocessing(img)
        kp, des = self.sift.detectAndCompute(img_pre, None)
        return kp, des

    def feature_matching_BF(self, kp, des, norm=cv.NORM_L2, k=2, dis_threshold=60, spatial_dis_threshold=10):
        bf = cv.BFMatcher(norm)
        matches = bf.knnMatch(des, des, k=k)
        good = []
        for m in matches:
            m = m[1:]
            for match in m:
                if match.distance < dis_threshold:
                    p1, p2 = np.array(kp[match.queryIdx].pt), np.array(
                        kp[match.trainIdx].pt)
                    if np.linalg.norm(p1-p2, ord=2) > spatial_dis_threshold:
                        good.append([p1, p2])
                else:
                    break
        return good


if __name__ == "__main__":
    df = utils.load_data_csv()
    img = cv.imread(os.path.join(utils.__rootdir__, df['img'][90]))
    sift = SIFT_Methods(utils.bgr2gray)
    kp, des = sift.feature_extraction(img)
    good = sift.feature_matching_BF(
        kp, des, k=3, dis_threshold=65, spatial_dis_threshold=10)
    for m in good:
        img = cv.line(img, m[0].astype(int), m[1].astype(int), (0, 255, 0), 1)
    # img = cv.drawKeypoints(
    #     img, good, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
