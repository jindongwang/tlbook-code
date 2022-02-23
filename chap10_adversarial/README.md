# Adversarial transfer learning

The code for adversarial transfer learning (DANN) is in `chap09_deeptransfer` since they share most parts of the code. Thus, you can go to that folder, and run the following command:

`python main.py --trans_loss dann --lamb 0.01`

By running the code, the result of amazon to webcam is 78.86%.

**NOTE:** To achieve better results, we have packaged them into a unified library [DeepDA](https://github.com/jindongwang/transferlearning/blob/17583db86d/code/DeepDA) with advanced training tricks. The results can be significantly improved. For instance, DANN can achieve **84.65%**. which is far better than 78.86% in this code. Feel free to explore DeepDA!