import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

import math
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse

from sklearn.model_selection import StratifiedKFold

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(args.length):
            _, img = cap.read()
            img = cv2.resize(img, (args.img, args.img))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def validation(model, criterion, val_loader, device, args):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

def train(model, optimizer, train_loader, val_loader, scheduler, device,args):
    model.to(device)
    if args.crash:
        criterion = nn.BCELoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_loss = math.inf
    best_model = None
    count = 0
    for epoch in range(1, args.epoch+1):
        model.train()
        train_loss = []
        count_label = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            count_label +=labels
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        
        if best_val_loss >= _val_loss:
            best_val_loss = _val_loss
            best_model = model
            torch.jit.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, args.save_name)
            count = 0
        else:
            count +=1
            if count > 10:
                print("Early Stopping")
                break
            
    return best_model    
    
def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    parser.add_argument("--device", type=int, default=0,help="select device number")
    parser.add_argument("--crash", action='store_true',help="Training about whether it crashs")
    parser.add_argument("--lr", type=float,default=1e-3, help="initial lr. Default 1e-3.")
    parser.add_argument("--seed", type=int,default=42, help="select random seed. Default 42.") 
    parser.add_argument("--epoch", type=int, default=200, help="train epoch. Default 200.")
    parser.add_argument("--kfold", type=int, default=3, help="kfold parameter. Default 5")
    parser.add_argument("--img", type=int, default=224, help="set input image size. Default 224")
    parser.add_argument("--batch", type=int, default=8, help="batch parameter. Default 8")
    parser.add_argument("--length" , type=int, default=50)
    parser.add_argument("--save_name", type=str, default=None, help="model name for save")
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(args.seed) # Seed 고정
    
    df = pd.read_csv("./data/train.csv")
    # weights sampler를 사용하기위해 shuffle= False
    skf = StratifiedKFold(n_splits=args.kfold,shuffle=False)
    for i,(train_index, val_index) in enumerate(skf.split(df,df['label'])):

        train_paths = df.loc[train_index]['video_path']
        train_labels = df.loc[train_index]['label']
        
        val_paths = df.loc[val_index]['video_path']
        val_labels = df.loc[val_index]['label']
        
        class_counts = train_labels.value_counts().to_dict()
        class_keys = sorted(class_counts.keys())
        num_samples = sum(class_counts.values())
        
        class_weights = [ round(1/class_counts[i],5) for i in train_labels.values]

        sampler = WeightedRandomSampler(torch.DoubleTensor(class_weights), num_samples)
        
        train_dataset = CustomDataset(train_paths.values,train_labels.values)
        train_loader = DataLoader(train_dataset, batch_size = args.batch,sampler=sampler)
        val_dataset = CustomDataset(val_paths.values,val_labels.values)
        val_loader = DataLoader(val_dataset, batch_size = args.batch, shuffle=False)
        
        model_name = "slowfast_r50"
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
        if args.crash:
            model.blocks.append(nn.Linear(400,1))
            model.blocks.append(nn.Sigmoid())
        else:
            model.blocks.append(nn.Linear(400,12))
            
        if args.save_name is None:
            if args.crash:
                save_name = "./checkpoint/crash"
            else:
                save_name = "./checkpoint/other"
                
            args.save_name = save_name + f"_{i}.pt"
        
        model.eval()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr,weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=args.lr*0.01)
        
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device,args)