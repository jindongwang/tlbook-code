# tlbook-code

This folder contains the codes for the book [Introduction to Transfer Learning: Algorithms and Practice](http://jd92.wang/tlbook). [迁移学习导论](http://jd92.wang/tlbook).

Links for the Chinese book (2nd edition) can be found at: [`links.md`](./links.md).

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

## Citation

If you find the code or the book helpful, please consider citing our book as:

```
@book{tlbook,
 author = {Wang, Jindong and Chen, Yiqiang},
 title = {Introduction to Transfer Learning},
 year = {2021},
 url = {jd92.wang/tlbook}
}

@book{tlbookchinese,
 author = {王晋东 and 陈益强},
 title = {迁移学习导论},
 year = {2021},
 url = {jd92.wang/tlbook}
}
```

## Recommended Repo

My unified transfer learning repo (and **the most popular** transfer learning repo on Github) has everything you need for transfer learning: https://github.com/jindongwang/transferlearning. Including: Papers, codes, datasets, benchmarks, applications etc. 
