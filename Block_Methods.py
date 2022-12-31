import cv2 as cv
import numpy as np

import utils


class Block_Detection_Base:
    def __init__(
        self,
        Block_Shape,
        Block_Stride,
        Preprocessing=None,
        Feature_Extraction=None,
        Matching=None
    ):
        self.Block_Shape = Block_Shape
        self.Block_Stride = Block_Stride
        self.Preprocessing = Preprocessing
        self.Feature_Extraction = Feature_Extraction
        self.Matching = Matching

    def cal_blocks(self, img):
        if self.Preprocessing is not None:
            img = self.Preprocessing(img)
        blocks = np.zeros((
            (img.shape[0]-self.Block_Shape[0]+1)//self.Block_Stride[0],
            (img.shape[1]-self.Block_Shape[1] + 1) // self.Block_Stride[1],
            self.Block_Shape[0],
            self.Block_Shape[1]
        ))
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                blocks[i, j] = img[i*self.Block_Stride[0]:i*self.Block_Stride[0] + self.Block_Shape[0],
                                   j*self.Block_Stride[1]:j*self.Block_Stride[1] + self.Block_Shape[1]]
        return blocks

    def cal_features(self, blocks):
        features = np.zeros((blocks.shape[0], blocks.shape[1], 7))
        for i, j in np.ndindex(blocks.shape[:2]):
            features[i, j] = self.Feature_Extraction(blocks[i, j])
        return features

    def cal_matching(self, features):
        matching = np.ones(
            (features.shape[0], features.shape[1], features.shape[0], features.shape[1]))*1000
        for i0, j0 in np.ndindex(matching.shape[:2]):
            for i1, j1 in np.ndindex(matching.shape[:2]):
                print(i0, j0, i1, j1)
                if i0 != i1 and j0 != j1:
                    matching[i0, j0, i1, j1] = self.Matching(
                        features[i0, j0], features[i1, j1])
        return matching


class Block_Detection_HuMoment(Block_Detection_Base):
    def __init__(
            self, Block_Shape, Block_Stride):
        super().__init__(Block_Shape, Block_Stride, utils.bgr2gray,
                         utils.HuMoment_Log, utils.Matching_Distance_L1)


if __name__ == "__main__":
    from Dataset import MICC_F220
    dataset = MICC_F220()
    img, label = dataset[0]
    bd = Block_Detection_HuMoment((32, 32), (32, 32))
    blocks = bd.cal_blocks(img)
    print(blocks.shape)
    features = bd.cal_features(blocks)
    print(features)
    matching = bd.cal_matching(features)
    print(matching)
    nodes = np.stack(np.where(matching < 1.5)).T
    print(nodes.shape[0])
    for x in range(nodes.shape[0]):
        cv.line(img, nodes[x, [1, 0]]*32+32,
                nodes[x, [3, 2]]*32+32, (0, 0, 255), 2)
    cv.imshow('matching', img)
    cv.waitKey(0)
    print('done')
