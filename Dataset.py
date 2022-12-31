import os

import cv2 as cv
from torch.utils.data import Dataset

import utils


class MICC_F220(Dataset):
    def __init__(self):
        self.df = utils.load_data_csv()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(utils.__rootdir__, self.df['img'][idx])
        image = cv.imread(img_path)
        label = self.df['label'][idx]

        return image, label


if __name__ == "__main__":
    dataset = MICC_F220()
    img, label = dataset[0]
    cv.imshow('img', img)
    cv.waitKey(0)
