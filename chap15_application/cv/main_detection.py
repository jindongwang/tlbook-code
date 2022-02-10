import os
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import xml.etree.ElementTree as ET
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.005)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--weight_decay', type=float, default='.0005')
    parser.add_argument('--data_path', type=str, default='./VOC/VOCdevkit/VOC2012')
    parser.add_argument('--no-pretrain', action='store_true')
    args = parser.parse_args()
    return args


class VOC(torch.utils.data.Dataset):
    def __init__(self, root_path, transforms):
        self.root =  root_path
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "JPEGImages"))))[:200]
        self.masks = list(sorted(os.listdir(os.path.join(self.root, "Annotations"))))[:200]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        annot_path = os.path.join(self.root, "Annotations", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        tree = ET.parse(annot_path)
        root = tree.getroot()

        boxes = []

        for neighbor in root.iter('bndbox'):
            xmin = int(neighbor.find('xmin').text)
            ymin = int(neighbor.find('ymin').text)
            xmax = int(neighbor.find('xmax').text)
            ymax = int(neighbor.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
        
        num_objs = len(boxes)
        

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model_detection(n_class, pretrain=True):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrain)
    num_classes = n_class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_data(root_path):
    dataset = VOC(root_path, get_transform(train=True))
    dataset_test = VOC(root_path, get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    loader_tr = torch.utils.data.DataLoader(
        dataset, batch_size=args.batchsize, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    loader_te = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batchsize * 2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    return loader_tr, loader_te

def train(loaders, model, device):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.nepoch):
        # train for one epoch (using pytorch's own function)
        train_one_epoch(model, optimizer, loaders['tr'], device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluation (also using pytorch's own function)
        evaluate(model, loaders['te'], device=device)
    
if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = args.nclass
    loader_tr, loader_te = load_data(args.data_path)
    loaders = {'tr': loader_tr, 'te': loader_te}
    model = get_model_detection(num_classes, not args.no_pretrain).to(device)
    train(loaders, model, device)