import argparse
import importlib

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import MICC_F220


def load_algorithm():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--algorithm',
        type=str,
        default="SIFT_Methods",
        help='algorithm code to test'
    )
    args = parser.parse_args()

    try:
        ALGO = importlib.import_module(args.algorithm)
        # Please make sure the algorithm class name is the same as the file name and has no arguments
        algorithm = getattr(ALGO, args.algorithm)()
    except:
        print('Error: Failed to load algorithm {}'.format(args.algorithm))
        exit(1)

    print("Successfully Load Algorithm {}".format(args.algorithm))
    return algorithm


def load_dataloader():
    dataset = MICC_F220()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    return dataloader


def result_evaluation(result):
    acc = (result[0, 0]+result[1, 1])/result.sum()
    prec = result[1, 1]/(result[1, 1]+result[0, 1])
    recall = result[1, 1]/(result[1, 1]+result[1, 0])
    F = 2*prec*recall/(prec+recall)
    return acc, prec, recall, F


def test_result(algorithm, dataloader):
    result = np.zeros((2, 2))
    for img, label in (pbar := tqdm(dataloader)):
        # print(img.shape)
        pred = algorithm.predict(img[0].numpy())
        result[label, pred] += 1
        acc, prec, recall, F = result_evaluation(result)
        pbar.set_description("Accuracy: {:.2f}% Precision:{:.2f}% Recall:{:.2f}% F Score: {:.2F}%".format(
            (acc*100), (prec*100), (recall*100), (F*100)))
        pbar.refresh()
    return result


if __name__ == "__main__":
    algorithm = load_algorithm()
    dataloader = load_dataloader()
    result = test_result(algorithm, dataloader)
