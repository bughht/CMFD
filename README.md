# CMFD

Copy-Move Forgery Detection and Localization: 
*This is a course project for media information security*

## Introduction

[Copy-Move Forgery Detection(CMFD)](https://en.wikipedia.org/wiki/Copy-move_forgery_detection) is a technique to detect and localize copy-move forgery in images. The goal of this project is to implement a CMFD algorithm in python and evaluate its performance.

## Dataset

[MICC-F220](http://lci.micc.unifi.it/labd/cmfd/MICC-F220.zip): this dataset is composed by 220 images; 110 are tampered and 110 originals.

## Project Goals

+ [x] Implement block-based algorithm.
+ [ ] Implement feature-point-based algorithm.
    + [x] SIFT feature
    + [ ] SURF feature
    + [ ] FAST feature
    + [ ] BRIEF feature
    + [ ] ORB feature
+ [ ] Implement matching algorithms
    + [x] Brute-force matching
    + [x] Fann matching
    + [ ] Hu moments matching
+ [ ] Design a model performance evaluation framework
  + [x] Torch Dataset and DataLoader wrapper
  + [ ] Model performance evaluation

## Contribution

We are welcome to any contribution to this project. If you are interested in this project, please contact us.

## References

+ Amerini, Irene, et al. "A sift-based forensic method for copy–move attack detection and transformation recovery." IEEE transactions on information forensics and security 6.3 (2011): 1099-1110.