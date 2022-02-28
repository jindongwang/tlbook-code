# tlbook-code

This folder contains the codes for the book [Introduction to Transfer Learning: Algorithms and Practice](http://jd92.wang/tlbook). [迁移学习导论](http://jd92.wang/tlbook).

## Dataset

1. For algorithm chapters (chapters 1 ~ 11), we mainly use Office-31 dataset, download [HERE](https://github.com/jindongwang/transferlearning/tree/master/data#office-31):
- For non-deep learning methods (chapters 1~7), we use ResNet-50 pre-trained features. Thus, download the ResNet-50 features.
- For deep learning methods (chapters 8~11), we use Office-31 original dataset. Thus, download the raw images.

2. For application chapters (chapters 15~19), the datasets download link can be found at respective chapters.

## Requirements

The following is a basic environment to run most experiments. No special tricky packages are needed. Just `pip install -r requirements.txt`.
- Python 3.x
- scikit-learn
- numpy
- scipy
- torch
- torchvision
