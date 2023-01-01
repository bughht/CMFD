import os

import cv2 as cv
import numpy as np

import utils


class FAST_Methods:
    def __init__(self, Preprocessing, threshold=45, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16):
        self.fast = cv.FastFeatureDetector_create(
            threshold=threshold, nonmaxSuppression=nonmaxSuppression, type=type)
        self.Preprocessing = Preprocessing

    def feature_extraction(self, img):
        """Extract features from preprocessed image

        Args:
            img (np.ndarray): input image of BGR

        Returns:
            kp, des: keypoints and descriptors
        """
        img_pre = self.Preprocessing(img)
        kp = self.fast.detect(img_pre, None)
        return kp

    def feature_matching_BF(self, kp, des, norm=cv.NORM_L2, k=2, dis_threshold=60, spatial_dis_threshold=10):
        """Function for feature matching using Brute Force 

        Args:
            kp (tuple): keypoints 
            des (np.ndarray): descriptors for keypoints, histogram of oriented gradients, shape (N, 128)
            norm (_type_, optional): _description_. Defaults to cv.NORM_L2.
            k (int, optional): _description_. Defaults to 2.
            dis_threshold (int, optional): _description_. Defaults to 60.
            spatial_dis_threshold (int, optional): _description_. Defaults to 10.

        Returns:
            good: Filtered matched points
        """
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

    def feature_matching_Flann(self, kp, des, dis_threshold=60, spatial_dis_threshold=10):
        """Function for feature matching using Flann

        Args:
            kp (tuple): keypoints 
            des (np.ndarray): descriptors for keypoints, histogram of oriented gradients, shape (N, 128)
            dis_threshold (int, optional): Feature distance threshold. Defaults to 60.
            spatial_dis_threshold (int, optional): Spatial distance threshold. Defaults to 10.

        Returns:
            good: Filtered matched points
        """
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des, des, k=2)
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
    img = cv.imread(os.path.join(utils.__rootdir__, df['img'][0]))

    fast = FAST_Methods(utils.bgr2gray)
    kp = fast.feature_extraction(img)
    img = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv.imshow('img', img)
    cv.waitKey(0)
