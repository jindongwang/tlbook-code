
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='imgs/model-g029de77ab_640.jpg')
    parser.add_argument('--tar_path', type=str, default='imgs/girl-gc797fc135_640.jpg')
    parser.add_argument('--nepoch', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--w_style', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--no-pretrain', action='store_true')
    args = parser.parse_args()
    return args

def load_images(src_path, tar_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
    img_src, img_tar = Image.open(src_path), Image.open(tar_path)
    img_src = img_src.resize((224, 224))
    img_tar = img_tar.resize((224, 224))
    img_src = transform(img_src).unsqueeze(0).cuda()
    img_tar = transform(img_tar).unsqueeze(0).cuda()
    return img_src, img_tar


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=not args.no_pretrain).features
        self.conv_layers = ['0', '5', '14', '19','28'] 
        
    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.conv_layers:
                features.append(x)
        return features

def train(model, imgs):
    img_src, img_tar = imgs['src'], imgs['tar']

    target = img_src.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([target], lr=args.lr, betas=[0.5, 0.999])
    for epoch in range(args.nepoch):

        fea_tar, fea_cont, fea_style = model(target), model(img_src), model(img_tar)

        loss_sty, loss_con = 0, 0
        for f_tar, f_con, f_sty in zip(fea_tar, fea_cont, fea_style):

            loss_con += torch.mean((f_tar - f_con)**2)

            _, c, h, w = f_tar.size()
            f_tar = f_tar.view(c, h * w)
            f_sty = f_sty.view(c, h * w)

            f_tar = torch.mm(f_tar, f_tar.t())
            f_sty = torch.mm(f_sty, f_sty.t())

            loss_sty += torch.mean((f_tar - f_sty)**2) / (c * h * w) 
        
        loss = loss_con + args.w_style * loss_sty 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % args.log_interval == 0:
            print(f'Epoch [{epoch+1}/{args.nepoch}], loss_con: {loss_con.item():.4f}, loss_sty: {loss_sty.item():.4f}')

        if (epoch+1) % args.sample_step == 0:

            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'pretrain-output-{}.png'.format(epoch+1))
    

if __name__ == "__main__":
    args = get_args()

    img_src, img_tar = load_images(args.src_path, args.tar_path)
    vgg = VGGNet().to(device).eval()
    train(vgg, {'src': img_src, 'tar': img_tar})
    
    