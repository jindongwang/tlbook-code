# Deep transfer learning demo

This demo code implements three different transfer learning algorithms:
- DDC: [Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/abs/1412.3474), published in arXiv 2014.
- DCORAL: [Deep CORAL Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf), published in ECCV 2016.
- DSAN: [Deep subdomain adaptation network for image classification](https://jd92.wang/assets/files/a24tnnls20.pdf), published in IEEE TNNLS 2020.

**Note:** This is only a demo code and the results are not heavily tuned. Also, we do not introduce many tricks such as learning rate scheduling.

## Requirement
* python 3
* pytorch 1.x
* torchvision

## Usage

1. You can download Office31 raw dataset [here](https://github.com/jindongwang/transferlearning/blob/17583db86d/data/readme.md#office-31).
2. After unziping the dataset, modify the folder name in `main.py` (Line 25).
3. Finally, you can directly run the code (or you can directly change the value in the code):
- DDC: `python main.py --trans_loss mmd --lamb 1`
- DCORAL: `python main.py --trans_loss coral --lamb 0.01`
- DSAN: `python main.py --trans_loss dsan --lamb 0.01`

## Results

By default running, we perform domain adaptation from amazon to webcam. The results are:
- DDC: 78.24%
- DCORAL: 79.00%
- DSAN: 79.24%

**NOTE:** To achieve better results, we have packaged them into a unified library [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA) with advanced training tricks. The results can be significantly improved. For instance, DSAN can achieve **94.34%**. which is far better than 79.24% in this code. Feel free to explore DeepDA!
