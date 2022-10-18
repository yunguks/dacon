import random
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm
from sklearn.model_selection import StratifiedKFold
import argparse

from myutils import make_class_info,cut_data,competition_metric,save_plot,sigmoid_focal_loss,CustomDataset

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class BaseModel(nn.Module):
    def __init__(self, num_classes=50,name='regnet'):
        super(BaseModel, self).__init__()
        if name =='convnext':
            self.backbone = timm.models.convnext.convnext_small(pretrained=True,num_classes=num_classes)
        elif name =='eiffcient':
            self.backbone = timm.models.efficientnet.efficientnet_b0(pretrained=True,num_classes=num_classes)
        else:
            self.backbone = timm.models.regnetx_064(pretrained=True,num_classes=num_classes)
        # self.classifier = nn.Linear(1000, num_classes)
        # self.drop = nn.Dropout(0.5,inplace=True)
    def forward(self, x):
        x = self.backbone(x)
        # x = self.drop(x)
        # x = self.classifier(x)
        return x

def validation(model,test_loader,criterion, device,classes):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label,classes)

            val_loss.append(loss.item())
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.argmax(1).detach().cpu().numpy().tolist()
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1

def train(model, optimizer, train_loader, criterion,scheduler,device,classes):
    model.train()
    train_loss = []
    for img , label in tqdm(iter(train_loader)):
        img, label = img.float().to(device), label.to(device)
        optimizer.zero_grad()

        model_pred = model(img)

        loss = criterion(model_pred, label,classes)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    if scheduler is not None:
        scheduler.step()

    tr_loss  = np.mean(np.array(train_loss))

    return tr_loss

def kfold_train(all_images, all_labels,class_info,opt,train_transform,test_transform=None,k=5,device='cpu'):
    print(f'Total Dataset : {all_images.shape}, {type(all_images)}')
    skf = StratifiedKFold(n_splits=k,shuffle=True,random_state=opt.seed)
    total_history = []
    data_length = len(all_images)//k
    print(data_length)
    c = 1
    for train_index, test_index in skf.split(all_images,all_labels):
        print(f'{c}st Train')
        train_images = all_images.iloc[train_index].values
        train_label = all_labels.iloc[train_index].values

        val_images = all_images.iloc[test_index].values
        val_label = all_labels.iloc[test_index].values

        train_dataset = CustomDataset(train_images,train_label,class_info,train_transform)
        train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle=True)

        val_dataset = CustomDataset(val_images,val_label,class_info,test_transform)
        val_loader = DataLoader(val_dataset, batch_size = opt.batch_size, shuffle=False)

        model = BaseModel(name=opt.model)
        model.to(device)

        best_loss = 100
        early = 0
        history = {'train_loss':[],'val_loss':[],'f1_score':[]}
        if opt.optim=='Adam':
            optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr_rate)
            scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)
            # scheduler = None
        else:
            optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=opt.lr_rate*0.01)

        criterion = sigmoid_focal_loss
        for e in range(1,opt.epochs):
            train_loss = train(model, optimizer, train_loader, criterion,scheduler, device, class_info)
            val_loss, f1_score = validation(model,val_loader, criterion,device, class_info)

            print(f'{e} epochs - T_loss : {train_loss:.5f}, V_loss : {val_loss:.5f}, F1 : {f1_score:.3f}')
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['f1_score'].append(f1_score)

            if best_loss> val_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },f'./checkpoint/best_{c}.pth')
                print('Model Saved')
                early=1
            else:
                early +=1
                if early > opt.stop:
                    print('Early stopping')
                    break
        total_history.append(history)
        c +=1
    return total_history

def inference(k,model, test_loader, device):
    model.to(device)
    model.eval()
    total = []
    with torch.no_grad():
        for i in range(k):
            checkpoint = torch.load(f'./checkpoint/best_{i+1}.pth')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f'{i+1}st inference ... ')

            model_pred = []
            for img in tqdm(iter(test_loader)):
                img = img.float().to(device)
                
                out = model(img)
                model_pred.append(out.detach().cpu().numpy().tolist())
            total.append(model_pred)
    result = []
    for j in range(len(total[0])):
        out = np.zeros(len(total[0]))
        for i in range(k):
            out += np.array(total[i][j])
        result.append(out.argmax(0))
    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--model',type=str, choices=['convnext','efficient','regent'], default='regent', help='[convnet, efficient, regent] choose one')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--k',type=int, default=5, help='k-fold parameter')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--lr-rate', type=float, default=5e-3)
    parser.add_argument('--optim', type=str, choices=['Adam','SGD'],default='SGD', help='[Adam, SGD] choose one')
    parser.add_argument('--stop', type=int, default=10, help='Early stopping count')
    parser.add_argument('--max-data', type=int, default=200, help='Up to a few per class')
    parser.add_argument('--device', type=int, choices=[0,1],default=0, help='choose 0 or 1 device')
    opt = parser.parse_args()

    GPU_NUM = opt.device # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    print(device)
    print(f'torch version : {torch.__version__}')

    seed_everything(opt.seed) # Seed 고정

    df = pd.read_csv('./Dataset/train.csv')
    new_df = pd.read_csv('./Dataset/artists_info.csv')

    # class별 정보, i번째 라벨은 label_to_name[i][0]
    class_info , label_to_name = make_class_info(df,new_df)
    
    df = cut_data(df,opt.max_data,class_info,opt.seed)

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224,224),
        A.HorizontalFlip( p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, p=0.5),
        # A.CoarseDropout(max_holes=4, max_height=16, max_width=16, 
        #                          min_holes=None, min_height=16, min_width=16,always_apply=False, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229,0.224,0.224), max_pixel_value=255),
        # (HxWxC) -> (CxHxW)
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229,0.224,0.224), max_pixel_value=255),
        ToTensorV2()
    ])
    
    result = kfold_train(df['img_path'], df['artist'],class_info,opt,train_transform,test_transform,k=opt.k,device=device)
    
    save_plot(result)
    
