import os

import cv2 as cv
import numpy as np
import pandas as pd
import torch

__rootdir__ = 'MICC-F220'


def load_data_csv():
    """Load Datapath and Label from groundtruthDB_220.txt

    Returns:
        pd.DataFrame: Dataframe with Datapath and Label
    """
    df = pd.read_csv(
        os.path.join(
            __rootdir__, 'groundtruthDB_220.txt'),
        header=None, sep='\s{4,}', engine='python')
    df.columns = ['img', 'label']
    return df


def bgr2gray(img):
    """Function for converting BGR image to GRAY image

    Args:
        img (np.ndarray): input BGR image

    Returns:
        np.ndarray: output GRAY image 
    """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def HuMoment_Log(img):
    """Function for calculating Hu Moment Log

    Args:
        img (np.ndarray): input image

    Returns:
        np.ndarray: Hu Moment Log
    """
    moments = cv.moments(img)
    hu = cv.HuMoments(moments).reshape(7)
    return -np.sign(hu)*np.log(np.abs(hu))


def Matching_Distance_L1(f1, f2):
    """Function for calculating L1 distance

    Args:
        f1 (np.ndarray): feature 1
        f2 (np.ndarray): feature 2

    Returns:
        float: L1 distance
    """
    return np.linalg.norm(f1-f2, ord=1)


def Matching_Distance_L2(f1, f2):
    """Function for calculating L2 distance

    Args:
        f1 (np.ndarray): feature 1
        f2 (np.ndarray): feature 2

    Returns:
        float: L2 distance
    """
    return np.linalg.norm(f1-f2, ord=2)


if __name__ == "__main__":
    df = load_data_csv()
    img = cv.imread(os.path.join(__rootdir__, df['img'][0]))
    print(img.shape)
    cv.imshow('img', img)
    cv.waitKey(0)
