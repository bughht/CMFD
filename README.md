# CMFD

Copy-Move Forgery Detection and Localization
*This is a course project for media information security*

## Introduction

Copy-Move Forgery Detection (CMFD) is a technique to detect and localize copy-move forgery in images. The goal of this project is to implement multiple CMFD algorithms in python and evaluate the performance.

We design a framework to evaluate the performance of the algorithm. The framework is based on PyTorch and can be easily extended to other algorithms.

Besides, we also implement a baseline algorithm (SIFT) and enhance it with patched self-adaptive methods to improve the performance.

## Group Info

+ [Haotian Hong](https://github.com/bughht)
+ [Zhenyu Jin](https://github.com/getOcr)

Phenomenon: *Talk is cheap, show me the code.*

## Dataset

[MICC-F220](http://lci.micc.unifi.it/labd/cmfd/MICC-F220.zip): this dataset is composed by 220 images; 110 are tampered and 110 originals.

## Pre-requisites

+ python>=3.7
+ opencv-python
+ numpy
+ sklearn
+ torch
+ pandas

## Installation

```bash
git clone https://github.com/bughht/CMFD.git
cd CMFD
wget http://lci.micc.unifi.it/labd/cmfd/MICC-F220.zip
unzip MICC-F220.zip
pip install -r Requirements.txt
```

## Usage

### Run the baseline

```bash
python AlgoTest.py -a SIFT_Methods
```

### Design your own algorithm

Make sure your algorithm is written in the format below:

Filename: `MyAlgorithm.py`

```python
class MyAlgorithm:
    def __init__(self, **kwargs):
        # initialize your algorithm

    def predict(self, img):
        # detect copy-move forgery in the image
        # return the classification result (0 or 1)
```

Then run the following command:

```bash
python AlgoTest.py -a MyAlgorithm
```

or

```bash
python AlgoTest.py --algorithm MyAlgorithm
```



## Baseline: SIFT

The baseline is a sift-based algorithm implemented in Python. With current parameters, the evaluation of this algorithm on MICC-F220 is shown below.

**Accuracy:** 81.36%
**Precision:** 76.74%
**Recall:** 90.00%
**F1 Score:** 82.85%

## Enhancement: Patch-SIFT

+ **Principle**: Split images into patches and adapt the parameters of SIFT (sigma) to the smoothness of the patch.
+ **Algorithm**:
  + Split the image into patches
  + For each patch, calculate the smoothness of the patch (using the variance of the Laplacian)
  + For each patch, adapt the parameters of SIFT (sigma) to the smoothness of the patch (using linear model) and apply SIFT to the patch
  + Apply Brute-Force Matching to the image
  + Evaluate the performance of the algorithm

**Accuracy:** 86.82%
**Precision:** 86.49%
**Recall:** 87.27%
**F1 Score:** 86.88%

## Experiment Results

We've tested the following algorithms on MICC-F220 dataset based on our framework:

+ Patch-SIFT
+ SIFT
+ ORB
+ FAST

## Patch-SIFT

| Patch-SIFT   | precision | recall | f1-score | support |
| ------------ | :-------- | :----- | :------- | :------ |
| No Copy-Move | 0.87      | 0.86   | 0.87     | 110     |
| Copy-Move    | 0.86      | 0.87   | 0.87     | 110     |

+ Accuracy:86.82% Precision:86.49% Recall:87.27%
+ Confusion Matrix

![cm](img/cm_PATCH_SIFT.png)

### SIFT

| SIFT         | precision | recall | f1-score | support |
| ------------ | :-------- | :----- | :------- | :------ |
| No Copy-Move | 0.88      | 0.73   | 0.80     | 110     |
| Copy-Move    | 0.77      | 0.90   | 0.83     | 110     |

+ Accuracy:81.36% Precision:76.74% Recall:90.00% F1 Score:82.85%
+ Confusion-Matrix:

![cm](img/cm_SIFT.png)

### ORB

| ORB          | precision | recall | f1-score | support |
| ------------ | :-------- | :----- | :------- | :------ |
| No Copy-Move | 0.65      | 0.75   | 0.69     | 110     |
| Copy-Move    | 0.70      | 0.60   | 0.65     | 110     |

+ Accuracy:67.27% Precision:70.21% Recall:60.00% F1 Score:64.71%
+ Confusion-Matrix:

![cm](img/cm_ORB.png)

## Project Goals Checkbox

+ [x] Implement feature-point-based algorithms
  + [x] Key Points Extraction
    + [x] SIFT
    + [x] ORB
    + [x] FAST
    + [x] Harris Corner
  + [x] Feature Descriptor
    + [x] SIFT feature
    + [x] ORB feature
+ [x] Implement matching algorithms
  + [x] Brute-force matching
  + [x] Fann matching
+ [x] Design a model performance evaluation framework
  + [x] Torch Dataset and DataLoader wrapper
  + [x] Model performance evaluation
+ [x] Enhance one of the algorithm tested above

## Contribution

We are welcome to any contribution to this project. If you are interested in this project, please contact us.

## References

<details>
<summary>
Expand all
</summary>
<pre>
[1]	FADL S M, SEMARY N A. Robust copy--move forgery revealing in digital images using polar coordinate system[J]. Neurocomputing, 2017, 265: 57-65.
[2]	LEE J C, CHANG C P, CHEN W K. Detection of copy--move image forgery using histogram of orientated gradients[J]. Information Sciences, 2015, 321: 250-262.
[3]	ULIYAN D M, JALAB H A, ABDUL WAHAB A W. Image region duplication forgery detection based on angular radial partitioning and Harris key-points[J]. Symmetry, 2016, 8(7): 62.
[4]	HOSNY K M, HAMZA H M, LASHIN N A. Copy-move forgery detection of duplicated objects using accurate PCET moments and morphological operators[J]. The Imaging Science Journal, 2018, 66(6): 330-345.
[5]	SZEGEDY C, LIU W, JIA Y. Going deeper with convolutions[J/OL]. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2015, 07-12-June: 1-9. DOI:10.1109/CVPR.2015.7298594.
[6]	KUZNETSOV A, MYASNIKOV V. A new copy-move forgery detection algorithm using image preprocessing procedure[J]. Procedia engineering, 2017, 201: 436-444.
[7]	PUN C M, CHUNG J L. A two-stage localization for copy-move forgery detection[J]. Information Sciences, 2018, 463: 33-55.
[8]	JWAID M F, BARASKAR T N. Detection of Copy-Move Image Forgery Using Local Binary Pattern with Discrete Wavelet Transform and Principle Component Analysis[C/OL]//2017 International Conference on Computing, Communication, Control and Automation (ICCUBEA). 2017: 1-6. DOI:10.1109/ICCUBEA.2017.8463695.
[9]	DIXIT R, NASKAR R, SAHOO A. Copy-move forgery detection exploiting statistical image features[C/OL]//2017 International Conference on Wireless Communications, Signal Processing and Networking (WiSPNET). 2017: 2277-2281. DOI:10.1109/WiSPNET.2017.8300165.
[10]	HILAL A, HAMZEH T, CHANTAF S. Copy-move forgery detection using principal component analysis and discrete cosine transform[C/OL]//2017 Sensors Networks Smart and Emerging Technologies (SENSET). 2017: 1-4. DOI:10.1109/SENSET.2017.8125021.
[11]	SÁNCHEZ J, MONZÓN N, SALGADO DE LA NUEZ A. An analysis and implementation of the harris corner detector[J]. Image Processing On Line, 2018.
[12]	LOWE D G. Distinctive image features from scale-invariant keypoints[J]. International journal of computer vision, 2004, 60: 91-110.
[13]	MUZAFFER G, ULUTAS G. A fast and effective digital image copy move forgery detection with binarized SIFT[C/OL]//2017 40th International Conference on Telecommunications and Signal Processing (TSP). 2017: 595-598. DOI:10.1109/TSP.2017.8076056.
[14]	SHAHROUDNEJAD A, RAHMATI M. Copy-move forgery detection in digital images using affine-SIFT[C/OL]//2016 2nd International Conference of Signal Processing and Intelligent Systems (ICSPIS). 2016: 1-5. DOI:10.1109/ICSPIS.2016.7869896.
[15]	JIN G, WAN X. An improved method for SIFT-based copy–move forgery detection using non-maximum value suppression and optimized J-Linkage[J]. Signal Processing: Image Communication, 2017, 57: 113-125.
[16]	LEE J C. Copy-move image forgery detection based on Gabor magnitude[J]. Journal of visual communication and image representation, 2015, 31: 320-334.
[17]	ZANDI M, MAHMOUDI-AZNAVEH A, TALEBPOUR A. Iterative Copy-Move Forgery Detection Based on a New Interest Point Detector[J/OL]. IEEE Transactions on Information Forensics and Security, 2016, 11(11): 2499-2512. DOI:10.1109/TIFS.2016.2585118.
[18]	AMERINI I, BALLAN L, CALDELLI R. A SIFT-Based Forensic Method for Copy–Move Attack Detection and Transformation Recovery[J/OL]. IEEE Transactions on Information Forensics and Security, 2011, 6(3): 1099-1110. DOI:10.1109/TIFS.2011.2129512.
[19]	YADAV N, KAPDI R. Copy move forgery detection using SIFT and GMM[C/OL]//2015 5th Nirma University International Conference on Engineering (NUiCONE). 2015: 1-4. DOI:10.1109/NUICONE.2015.7449647.
[20]	ALBERRY H A, HEGAZY A A, SALAMA G I. A fast SIFT based method for copy move forgery detection[J]. Future Computing and Informatics Journal, 2018, 3(2): 159-165.
[21]	MOUSSA A M. A fast and accurate algorithm for copy-move forgery detection[C/OL]//2015 Tenth International Conference on Computer Engineering & Systems (ICCES). 2015: 281-285. DOI:10.1109/ICCES.2015.7393060.
[22]	LI J, LI X, YANG B. Segmentation-based image copy-move forgery detection scheme[J]. IEEE transactions on information forensics and security, 2014, 10(3): 507-518.
[23]	WANG X Y, LI S, LIU Y N. A new keypoint-based copy-move forgery detection for small smooth regions[J]. Multimedia Tools and Applications, 2017, 76: 23353-23382.
[24]	FISCHLER M A, BOLLES R C. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography[J]. Communications of the ACM, 1981, 24(6): 381-395.
[25]	AL-HAMMADI M M, EMMANUEL S. Improving SURF Based Copy-Move Forgery Detection Using Super Resolution[C/OL]//2016 IEEE International Symposium on Multimedia (ISM). 2016: 341-344. DOI:10.1109/ISM.2016.0075.
[26]	COZZOLINO D, POGGI G, VERDOLIVA L. Efficient Dense-Field Copy–Move Forgery Detection[J/OL]. IEEE Transactions on Information Forensics and Security, 2015, 10(11): 2284-2297. DOI:10.1109/TIFS.2015.2455334.


</pre>
</details>
