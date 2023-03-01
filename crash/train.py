import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import CustomDataset, seed_everything, collate_fn

import albumentations as A

import math
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse

from sklearn.model_selection import StratifiedKFold
from collections import Counter
    
def validation(model, criterion, val_loader, device, args):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in iter(val_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            if args.crash:
                preds += (logit > 0.5).squeeze(-1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()
            else:
                preds += logit.argmax(1).detach().cpu().numpy().tolist()
                trues += labels.argmax(1).detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

def train(model, optimizer, train_loader, val_loader, scheduler, device,args):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    best_val_loss = math.inf
    best_model = None
    count = 0
    for epoch in range(1, args.epoch+1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
            train_loss.append(loss.item())
            
        _val_loss, _val_score = validation(model, criterion, val_loader, device,args)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        
        if best_val_loss >= _val_loss:
            best_val_loss = _val_loss
            best_model = model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, args.save_name)
            print(f"save model {args.save_name}")
            count = 0
        else:
            count +=1
            if count > 10:
                print("Early Stopping")
                break
            
    return best_model    
    
def get_args():
    parser = argparse.ArgumentParser(description="Crach training")
    parser.add_argument("--device", type=int, default=0,help="select device number")
    # model
    parser.add_argument("--crash", action='store_true',help="Training about whether it crashs")
    parser.add_argument("--weather", action="store_true", help="Using label to binary list")
    parser.add_argument("--small", action="store_true", help="using x3d_xs model. default is x3d_l")
    
    # train
    parser.add_argument("--lr", type=float,default=1e-3, help="initial lr. Default 1e-3.")
    parser.add_argument("--seed", type=int,default=42, help="select random seed. Default 42.") 
    parser.add_argument("--epoch", type=int, default=200, help="train epoch. Default 200.")
    parser.add_argument("--kfold", type=int, default=5, help="kfold parameter. Default 5")
    parser.add_argument("--img", type=int, default=224, help="set input image size. Default 224")
    parser.add_argument("--batch", type=int, default=6, help="batch parameter. Default 8")
    parser.add_argument("--length" , type=int, default=50)
    
    # utils
    parser.add_argument("--save_name", type=str, default=None, help="model name for save")
    parser.add_argument("--txt", type=str, default=None, help="output metrix save in txt")
    return parser.parse_args()

if __name__=="__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = get_args()
    print(args)
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(args.seed) # Seed 고정
    
    df = pd.read_csv("./data/train.csv")
    # weights sampler를 사용하기위해 shuffle= False
    if args.crash is False:
        df = df[df['label']>0]
        df.reset_index(drop=True,inplace=True)

    skf = StratifiedKFold(n_splits=args.kfold,shuffle=False)
    for k,(train_index, val_index) in enumerate(skf.split(df,df['label'])):
        train_paths = df.iloc[train_index]['video_path'].values
        train_labels = df.iloc[train_index]['label'].values
   
        val_paths = df.iloc[val_index]['video_path'].values
        val_labels = df.iloc[val_index]['label'].values
     
        # 클래스별 개수를 구하여 sampling
        class_counts = Counter(train_labels)
        weights = torch.DoubleTensor([1./class_counts[i] for i in train_labels])
        weight_sampler = WeightedRandomSampler(weights,len(train_labels))
        
        train_transforms = A.Compose([
            A.Resize(height=180, width=320),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(max_pixel_value=255.0)
        ],additional_targets={f"image{i}":"image" for i in range(1, 50)})
        
        val_transforms = A.Compose([
            A.Resize(height=180, width=320),
            A.Normalize(max_pixel_value=255.0)
        ],additional_targets={f"image{i}":"image" for i in range(1, 50)})
        
        train_dataset = CustomDataset(train_paths,train_labels,args,train_transforms)
        # collate_fn -> mixup
        train_loader = DataLoader(train_dataset, batch_size = args.batch,sampler=weight_sampler,collate_fn=collate_fn)
        
        val_dataset = CustomDataset(val_paths,val_labels,args,val_transforms)
        val_loader = DataLoader(val_dataset, batch_size = args.batch, shuffle=False)
        
        if args.small:
            model_name = "x3d_s"
        else:
            model_name = "x3d_s"
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
        if args.crash or args.weather:
            model.blocks.append(nn.Dropout(0.3))
            model.blocks.append(nn.Linear(400,3))
        else:
            model.blocks.append(nn.Linear(400,1))
            
        if args.save_name is None:
            if args.crash:
                save_name = "./checkpoint/crash"
            elif args.weather:
                save_name = "./checkpoint/weather"
            else:
                save_name = "./checkpoint/time"
                
        args.save_name = save_name + f"_{k+1}.pt"

        model.eval()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr,weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=args.lr*0.01)
        
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device,args)