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
from collections import Counter
    
class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list,args):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.args = args
        
        key ,class_sample_count = torch.unique(torch.tensor(self.label_list), return_counts=True)
        print(key)
        print(class_sample_count)
        # 데이터 샘플링 가중치 계산
        weights = 1. / class_sample_count.float()
        print(weights)
        if self.args.crash:
            self.weights = weights[self.label_list]
        else:
            self.weights = weights[self.label_list-1]
        
    def __getitem__(self, index):
        frames = self.get_video("./data"+self.video_path_list[index][1:])
        
        if self.label_list is not None:
            label = self.label_list[index]
            if self.args.crash:
                if label!=0:
                    label = 1
                label = torch.FloatTensor([label])
            else:
                label -=1
            
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
            img = cv2.resize(img, (self.args.img, self.args.img))
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
            
            if args.crash:
                preds += (logit > 0.5).squeeze(-1).detach().cpu().numpy().tolist()
            else:
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
        count_c = []
        for videos, labels in tqdm(iter(train_loader)):
            count_c += labels.squeeze(-1).detach().cpu().numpy().tolist()
            continue
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
        
        print(Counter(count_c))
        continue
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
    print(args)
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(args.seed) # Seed 고정
    
    df = pd.read_csv("./data/train.csv", index_col=None)
    # weights sampler를 사용하기위해 shuffle= False
    if args.crash:
        df['label']=np.where(df['label'] >0, 1,0)
    else:
        df = df[df['label']>0]
        df.reset_index(drop=True,inplace=True)

    skf = StratifiedKFold(n_splits=args.kfold,shuffle=False)
    for k,(train_index, val_index) in enumerate(skf.split(df,df['label'])):
        train_paths = df.iloc[train_index]['video_path'].values
        train_labels = df.iloc[train_index]['label'].values
        
        val_paths = df.iloc[val_index]['video_path'].values
        val_labels = df.iloc[val_index]['label'].values
        # 0을 제외한 클래스 개수를 계산합니다.
        class_counts = Counter(train_labels)
        class_keys = sorted(class_counts.keys())

        # 0을 제외한 클래스 별 가중치를 계산합니다.
        class_weights = [1.0 / class_counts[i] for i in class_keys]
        weights = torch.DoubleTensor([class_weights[i - 1] for i in train_labels])
        sampler = WeightedRandomSampler(weights, len(weights))
    
        train_dataset = CustomDataset(train_paths,train_labels,args)
        # sampler = WeightedRandomSampler(train_dataset.weights, int(len(train_labels)))
        train_loader = DataLoader(train_dataset, batch_size = args.batch,sampler=sampler)
        
        val_dataset = CustomDataset(val_paths,val_labels,args)
        val_loader = DataLoader(val_dataset, batch_size = args.batch, shuffle=False)

        model_name = "x3d_xs"
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
                
            args.save_name = save_name + f"_{k}.pt"
        
        model.eval()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr,weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=args.lr*0.01)
        
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device,args)