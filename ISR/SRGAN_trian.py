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
    def __init__(self, img_list, transform=None, train_mode=True):
        self.transform = transform
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
            if self.transform is not None:
                high_img = self.transform(high_img)
            return low_img, high_img
        else:
            file_name = low_path.split('/')[-1]
                
            return low_img, file_name
    
    def __len__(self):
        return len(self.low_list)

def train(train_loader,val_loader, model, epochs,criterion, optimizer, scheduler=None, device='cpu'):
    model.to(device)
    best_loss = 9999
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []
    count = 0
    for epoch in range(1,epochs+1):
        model.train()
        running_loss =0.0

        for data in tqdm(iter(train_loader)):
            lr_img = data[0].to(device)
            hr_img = data[1].to(device)

            optimizer.zero_grad()

            outputs = model(lr_img)
            loss = criterion(outputs,hr_img)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
        
        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(iter(val_loader)):
                ir_img = data[0].to(device)
                hr_img = data[1].to(device)

                outputs = model(ir_img)
                loss = criterion(outputs,hr_img)

                val_loss +=loss.item()
        print(f'{epoch} Train Loss : {running_loss:.5f}')
        print(f'val_loss : {val_loss:.5f}')
        
        # early stopping 
        if best_loss <= val_loss:
            count +=1
            if count >4:
                print('Early Stopping')
                break
        else:
            torch.save(model.state_dict(),f'checkpoint/best_SRCNN2.pth')
            print('Model Saved')
            best_loss = val_loss
            count = 1

        history['train_loss'].append(running_loss)
        history['val_loss'].append(val_loss)
    return history

def inference(model, test_loader ,device):
    model.to(device)
    model.eval()
    pred_img_list = []
    name_list = []
    with torch.no_grad():
        for lr_img, file_name in tqdm(iter(test_loader)):
            lr_img = lr_img.float().to(device)

            pred_img = model(lr_img)

            for pred, name in zip(pred_img,file_name):
                pred = pred.cpu().clone().detach().numpy()
                pred = pred.transpose(1,2,0)
                pred = pred*255

                pred_img_list.append(pred.astype('uint8'))
                name_list.append(name)
    return pred_img_list,name_list


if __name__ =='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    torch.cuda.empty_cache()
    gc.collect()

    csv_path = os.getcwd()+'/data/train.csv'
    df = pd.read_csv(csv_path)

    BATCH_SIZE =4
    IMG_SIZE=2048
    EPOCH=200

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([IMG_SIZE,IMG_SIZE]),
        torchvision.transforms.ToTensor(),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([IMG_SIZE,IMG_SIZE]),
        torchvision.transforms.ToTensor(),
    ])

    train_path = os.getcwd()+'/data/train.csv'
    train_list = pd.read_csv(train_path)
    train_list = train_list[0:int(len(train_list)*0.75)]
    val_list = train_list[int(len(train_list)*0.75):]

    train_dataset = CustomDataset(train_list,train_transform,train_mode=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
        )

    val_dataset = CustomDataset(val_list,test_transform,train_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,BATCH_SIZE,False)


    test_path = os.getcwd()+'/data/test.csv'
    test_list = pd.read_csv(test_path)
    test_dataset = CustomDataset(test_list,test_transform,train_mode=False)
    test_loader = DataLoader(test_dataset,BATCH_SIZE,False)

    netG = SRGAN.Generator
    netD = SRGAN.Discriminator

    criterion = SRGAN.GeneratorLoss()
    optimizerG = torch.optim.SGD(params=netG.parameters(), lr =0.01,momentum=0.9)
    optimizerD = torch.optim.SGD(params=netD.parameters(), lr =0.01,momentum=0.9)

    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG,step_size= 5,gamma=0.3)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD,step_size= 5,gamma=0.3)

    history = {'d_loss':[], 'g_loss':[],'d_score':[],'g_score':[],'psnr':[],'ssim':[]}
    best_psnr = 0.0
    count =1
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
            fake_out = netD(fake_img).mean()
            d_loss = 1-real_out + fake_out
            # 기울기가 연산하고 없어지지 않도록
            d_loss.backward(retain_graph=True)
            optimizerD.step()

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
            running_d_loss +=d_loss.item() 
        print(f'{epoch} Train Loss G : {running_g_loss:.5f}, Loss D : {running_d_loss:.5f}')
        
        # val data
        netG.eval()
        with torch.no_grad():
            for val_lr, val_hr in tqdm(iter(val_loader)):
                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)

                sr = netG(val_lr)
                
                mse = ((sr-val_hr)**2).data.mean()
                ssim = pytorch_ssim.ssim(sr,val_hr).item()
                psnr = 10*math.log10((val_hr.max()**2)/mse)
        history['d_loss'].append(running_d_loss)
        history['g_loss'].append(running_g_loss)
        history['psnr'].append(psnr)
        history['ssim'].append(ssim)

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

    checkpoint = torch.load(f'checkpoint/best_netG_{best_psnr}.pth')
    netG.load_state_dict(checkpoint, strict=False)

    pred_img_list, pred_name_list = inference(netG, test_loader,device)

    os.chdir('/workspace/Torch/dacon/ISR/data/submission/')
    # os.chdir('./data/submission/')
    sub_imgs = []
    for path, pred_img in tqdm(zip(pred_name_list,pred_img_list)):
        cv2.imwrite(path,pred_img)
        sub_imgs.append(path)

    submission = zipfile.ZipFile('../submission_SRGAN.zip','w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()
    print('Done')
