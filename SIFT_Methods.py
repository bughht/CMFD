import os

import cv2 as cv
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import utils


class SIFT_Methods:
    def __init__(self, Preprocessing):
        self.sift = cv.SIFT_create(
            nfeatures=0, nOctaveLayers=3, contrastThreshold=0.03, edgeThreshold=80, sigma=1.3)
        self.Preprocessing = Preprocessing

    def feature_extraction(self, img):
        """Extract features from preprocessed image

        Args:
            img (np.ndarray): input image of BGR

        Returns:
            kp, des: keypoints and descriptors
        """
        img_pre = self.Preprocessing(img)
        kp, des = self.sift.detectAndCompute(img_pre, None)
        return kp, des

    def feature_matching_BF(self, kp, des, norm=cv.NORM_L2, k=3, dis_threshold=60, spatial_dis_threshold=10):
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
        return np.asarray(good)

    def feature_matching_Flann(self, kp, des, k=2, dis_threshold=60, spatial_dis_threshold=10):
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
        matches = flann.knnMatch(des, des, k=k)
        pair = []
        for m in matches:
            m = m[1:]
            for match in m:
                if match.distance < dis_threshold:
                    p1, p2 = np.array(kp[match.queryIdx].pt), np.array(
                        kp[match.trainIdx].pt)
                    if np.linalg.norm(p1-p2, ord=2) > spatial_dis_threshold:
                        pair.append([p1, p2])
                else:
                    break
        return np.asarray(pair)

    def predict(self,
                img,
                dis_threshold=50,
                spatial_dis_threshold=5
                ):

        kp, des = self.feature_extraction(img)
        matchpt = self.feature_matching_BF(
            kp, des, dis_threshold=dis_threshold, spatial_dis_threshold=spatial_dis_threshold)
        if matchpt.shape[0] == 0:
            return 0
        else:
            pts = np.unique(np.vstack(matchpt), axis=0)
            dist_matrix = pdist(pts, metric='euclidean')
            Z = hierarchy.linkage(dist_matrix, metric='euclidean')
            cluster = hierarchy.fcluster(
                Z, t=2, criterion='inconsistent', depth=4)
            _, counts = np.unique(cluster, return_counts=True)
            counts = np.sort(counts)
            top2_cluster = counts[-2:]
            if top2_cluster.sum() >= 8:
                return 1
            else:
                return 0


if __name__ == "__main__":
    from time import time
    df = utils.load_data_csv()
    img = cv.imread(os.path.join(utils.__rootdir__, df['img'][210]))
    sift = SIFT_Methods(utils.bgr2gray)
    t0 = time()
    pred = sift.predict(img)
    t1 = time()
    print("Prediction: {}".format(pred))
    print("Time Cost: {}".format(t1-t0))
    kp, des = sift.feature_extraction(img)
    matchpt = sift.feature_matching_Flann(
        kp, des, dis_threshold=55, spatial_dis_threshold=5)
    # print(matchpt.shape)
    # pts = np.unique(np.vstack(matchpt), axis=0)
    # obtain condensed distance matrix (needed in linkage function)
    for m in matchpt:
        img = cv.line(img, m[0].astype(int), m[1].astype(int), (0, 255, 0), 1)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
