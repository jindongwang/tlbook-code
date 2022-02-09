# Code for cross-domain activity recognition


## Usage

Download the dataset at https://www.kaggle.com/jindongwang92/crossposition-activity-recognition.

### Source selection and classification

Currently, we only support DSADS dataset.

For RA as target: `cd select_classify; python select_classify.py --target RA --k 1`

### Deep transfer learning for classification

For RA-LA: `cd deep_transfer_har; python deep_transfer_har.py --lr 0.005 --batchsize 128 --lamb 5 --loss mmd`


### Reference

```
@inproceedings{wang2018deep,
  title={Deep transfer learning for cross-domain activity recognition},
  author={Wang, Jindong and Zheng, Vincent W and Chen, Yiqiang and Huang, Meiyu},
  booktitle={proceedings of the 3rd International Conference on Crowd Science and Engineering},
  pages={1--8},
  year={2018}
}
```