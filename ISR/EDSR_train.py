import pandas as pd
import numpy as np
import torch
import torchvision
import os
import cv2
import einops
import random
from torch.utils.data import DataLoader
from torch import nn

GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
os.environ['CUDA_LAUNCH_BLOCKING']="1"
print(device)
print(f'torch version : {torch.__version__}')

import math
def cal_psnr(img1,img2):
    PIXEL_MAX = 1.0
    # # input shaep가 [ num_patch, 3, 64, 64] 일경우
    #p = 0
    #for i in range(img1.shape[0]):
        #PSNR구하는 코드
    #    p += 10 * math.log10(PIXEL_MAX / np.mean((x[i] - y[i]) ** 2) )
    #return p/img1.shape[0]

    # input shape가 [3, 64, 64] 일 경우
    return 10 * math.log10(PIXEL_MAX / np.mean((img1 - img2) ** 2) )

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None, train_mode=True):
        self.transform = transform
        self.train_mode = train_mode
        self.low_list = img_list['LR']
        if self.train_mode:
            self.high_list = img_list['HR']
        self.patch_size = 256//4

    def __getitem__(self,index,check=False):
        h = random.randrange(255-self.patch_size)
        w = random.randrange(255-self.patch_size)
        low_path = './Dataset'+self.low_list.iloc[index][1:]
        low_img = cv2.imread(low_path)

        low_img = low_img[h:h+self.patch_size,w:w+self.patch_size, :]

        if low_img is None:
            print(f'{low_path} is exist? {os.path.isfile(low_path)}')
        if check:
            print(f'image path : {low_path}')
            print(f'image shape : {low_img.shape}')
            print(f'image type : {type(low_img)}')

        if self.transform is not None:
            torch.manual_seed(41)
            low_img=self.transform(low_img)

        
        if self.train_mode:
            high_path = './Dataset'+self.high_list.iloc[index][1:]
            high_img = cv2.imread(high_path)
            high_img = high_img[4*h:4*h+self.patch_size*4, 4*w:4*w+self.patch_size*4, :]
            if self.transform is not None:
                torch.manual_seed(41)
                high_img = self.transform(high_img)
            return low_img, high_img
        
        file_name = low_path.split('/')[-1]    
        return low_img, file_name
    
    def __len__(self):
        return len(self.low_list)

class test_CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None, train_mode=False):
        self.transform = transform
        self.low_list = img_list['LR']
        self.train_mode = train_mode
        if self.train_mode:
            self.high_list = img_list['HR']

    def __getitem__(self,index,check=False):
        low_path = './Dataset'+self.low_list.iloc[index][1:]
        low_img = cv2.imread(low_path)
        if check:
            print(f'image path : {low_path}')
            print(f'image shape : {low_img.shape}')
            print(f'image type : {type(low_img)}')

        if self.transform is not None:
            torch.manual_seed(41)
            low_img=self.transform(low_img)

        if self.train_mode:
            high_path = './Dataset'+self.high_list.iloc[index][1:]
            high_img = cv2.imread(high_path)
            if self.transform is not None:
                torch.manual_seed(41)
                high_img=self.transform(high_img)
            
            return low_img, high_img

        file_name = low_path.split('/')[-1]
                
        return low_img, file_name

    def __len__(self):
        return len(self.low_list)

seed_everything(41)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    #torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
])

from torch import nn
from einops.layers.torch import Rearrange

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class Upsampler(nn.Sequential):
    def __init__(self, n_feats, bn=False, act=False, bias=True):
        m = []
        for _ in range(2):
            m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, stride=1, padding =1, bias=bias))
            m.append(nn.PixelShuffle(2))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super(ResBlock, self).__init__()

        self.act = nn.ReLU(True)
        self.res_scale = res_scale

        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3,padding=1,bias=True))
            if i == 0:
                m.append(self.act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, scale_factor=2*2, num_channels=3, num_feats=256, num_blocks=32, res_scale=0.1, img_mean=(0.4411, 0.479, 0.4414)):
        super(EDSR, self).__init__()

        self.sub_mean = MeanShift(img_mean, sub=True)
        self.add_mean = MeanShift(img_mean, sub=False)

        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3,bias=True, padding=3//2)
        body = [ResBlock(num_feats, res_scale) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_feats, num_feats, kernel_size=3,padding=1))
        self.body = nn.Sequential(*body)

        tail = [
            Upsampler(num_feats),
            nn.Conv2d(num_feats, num_channels, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        self.tail = nn.Sequential(*tail)
        self.initialize_weights()

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

BATCH_SIZE =16

train_list = pd.read_csv('./Dataset/train.csv')
train_list = train_list[int(len(train_list))*0.2:]
val_list = train_list[:int(len(train_list))*0.2]
print(f'train data : {len(train_list)}, val data : {len(val_list)}')

train_dataset = CustomDataset(train_list,train_transform,train_mode=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
    )

val_dataset = test_CustomDataset(val_list,test_transform,train_mode=True)
val_loader = torch.utils.data.DataLoader(val_dataset,1,False)

test_list = pd.read_csv('./Dataset/test.csv')
print(f'test data : {len(test_list)}')
test_dataset = CustomDataset(test_list,test_transform,train_mode=False)
test_loader = DataLoader(test_dataset,BATCH_SIZE,False)

model = EDSR()
load_model = torch.load('./checkpoint/edsr_x4-4f62e9ef.pt',map_location=device)
model.load_state_dict(load_model, strict=False)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=2,eta_min=1e-7)
scheduler = None
print('parameter set')

from tqdm import tqdm

history = {'loss':[],'val_psnr':[]}
best_psnr = 0.0
count =1
EPOCH = 1000

print('start train')
torch.cuda.empty_cache()
model.to(device)
print(torch.cuda.memory_allocated() / 1024 /1024)
for epoch in range(1, EPOCH+1):
    running_loss = 0

    model.train()
    for lr, hr in tqdm(iter(train_loader)):
        # lr.shape [1,512//64 * 512//64 , 3, 64,64]
        lr_img = lr.to(device)
        hr_img = hr.to(device)

        optimizer.zero_grad()
        out = model(lr_img)

        loss = criterion(out,hr_img)

        loss.backward()
        running_loss +=loss.item()
        optimizer.step()
    print(f'{epoch} Train Loss : {running_loss/len(train_loader):5f}')
    
    # val data
    if epoch %20 ==0:
        model.eval()
        psnr = 0
        with torch.no_grad():
            # 한 이미지씩 패치로
            for lr_img, hr_img in tqdm(iter(val_loader)):
                lr_img = lr_img.squeeze(0)
                hr_img = hr_img.squeeze(0)
                lr_img = einops.rearrange(lr_img, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=64,p2=64)
                pred_hr_img = []
                # [64 , 3, 64, 64]
                for i in range(lr_img.shape[0]):
                    val_lr = lr_img[i].to(device)
                    out = model(val_lr)

                    pred_hr_img.append(out.detach().cpu().numpy())
                # [64, 3, 256, 256]
                pred = einops.rearrange(pred_hr_img, '(h w) c p1 p2 -> c (h p1) (w p2)', h=8, w=8)

                psnr += cal_psnr(pred,hr_img.numpy())

        psnr = round(psnr/len(val_loader),5)
        print(f'Val psnr : {psnr}')

        history['loss'].append(running_loss)
        history['val_psnr'].append(psnr)


        if best_psnr < psnr:
            best_psnr = psnr
            count = 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },f'./checkpoint/best_EDSR.pth')
            print('Model Saved')
        else:
            if count > 10:
                print('Early Stopping')
                break
            count +=1
    
