import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from tqdm.auto import tqdm
import PIL
import gc
import zipfile
from models import SRGAN
import pytorch_ssim
import math

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None,hr_transform=None, train_mode=True):
        self.transform = transform
        self.hr_transform = hr_transform
        self.train_mode = train_mode

        self.low_list = img_list['LR']
        if self.train_mode:
            self.high_list = img_list['HR']

    def __getitem__(self,index):
        low_path = './data/'+self.low_list.iloc[index][1:]
        low_img = PIL.Image.open(low_path)
        
        if self.transform is not None:
            low_img=self.transform(low_img)
        
        if self.train_mode:
            high_path = './data/'+self.high_list.iloc[index][1:]
            high_img = PIL.Image.open(high_path)
            if self.hr_transform is not None:
                high_img = self.hr_transform(high_img)
            return low_img, high_img
        else:
            file_name = low_path.split('/')[-1]
                
            return low_img, file_name
    
    def __len__(self):
        return len(self.low_list)

if __name__ =='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    print(device)
    print(f'torch version : {torch.__version__}')
    #torch.cuda.empty_cache()
    #gc.collect()

    BATCH_SIZE =1
    IMG_SIZE=512
    EPOCH=200

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([IMG_SIZE,IMG_SIZE]),
        torchvision.transforms.ToTensor(),
    ])
    train_hr_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_list = pd.read_csv('./data/train.csv')
    train_list = train_list[0:int(len(train_list)*0.75)]
    val_list = train_list[int(len(train_list)*0.75):]

    train_dataset = CustomDataset(train_list,train_transform,train_hr_transform,train_mode=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
        )

    val_dataset = CustomDataset(val_list,train_transform,train_hr_transform,train_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,BATCH_SIZE,False)
    print('data load')
    netG = SRGAN.Generator(4)
    netD = SRGAN.Discriminator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    print('model set')
    criterion = SRGAN.GeneratorLoss().to(device)
    optimizerG = torch.optim.SGD(params=netG.parameters(), lr =0.1,momentum=0.9)
    optimizerD = torch.optim.SGD(params=netD.parameters(), lr =0.1,momentum=0.9)

    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG,step_size= 5,gamma=0.3)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD,step_size= 5,gamma=0.3)
    print('parameter set')
    history = {'d_loss':[], 'g_loss':[],'d_score':[],'g_score':[],'psnr':[],'ssim':[]}
    best_psnr = 0.0
    count =1
    print('start train')
    for epoch in range(1, EPOCH+1):
        netG.to(device)
        netD.to(device)
        netG.train()
        netD.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        for lr_img, hr_img in tqdm(iter(train_loader)):
            lr_img= lr_img.to(device)
            hr_img= hr_img.to(device)

            # update Net D
            fake_img = netG(lr_img)
            netD.zero_grad()
            real_out = netD(hr_img).mean()
            print(f'real_out.shape :{real_out.shape}')
            fake_out = netD(fake_img).mean()
            print(f'fake_out.shape :{fake_out.shape}')
            d_loss = 1-real_out + fake_out
            del real_out
            # 기울기가 연산하고 없어지지 않도록
            d_loss.backward(retain_graph=True)
            optimizerD.step()
            running_d_loss +=d_loss.item() 

            # update Net G
            netG.zero_grad()
            fake_img = netG(lr_img)
            fake_out = netD(fake_img).mean()
            g_loss = criterion(fake_out,fake_img,hr_img)
            g_loss.backward()

            fake_img = netG(lr_img)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            running_g_loss +=g_loss.item() 

        print(f'{epoch} Train Loss G : {running_g_loss:.5f}, Loss D : {running_d_loss:.5f}')
        
        # val data
        netG.eval()
        mse = 0
        ssim = 0
        psnr = 0
        with torch.no_grad():
            for val_lr, val_hr in tqdm(iter(val_loader)):
                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)

                sr = netG(val_lr)

                mse += ((sr-val_hr)**2).data.mean()
                ssim += pytorch_ssim.ssim(sr,val_hr).item()
                psnr += 10*math.log10((val_hr.max()**2)/mse).cpu().numpy()

        history['d_loss'].append(running_d_loss)
        history['g_loss'].append(running_g_loss)
        history['psnr'].append(psnr/len(val_loader))
        history['ssim'].append(ssim.cpu().numpy()/len(val_loader))

        if best_psnr < psnr:
            torch.save(netG.state_dict(),f'checkpoint/best_netG_{psnr}.pth')
            torch.save(netD.state_dict(),f'checkpoint/best_netD_{psnr}.pth')
            print('Model Saved')
            best_psnr = psnr
            count = 1
        else:
            if count > 5:
                print('Early Stopping')
                break
            count +=1
