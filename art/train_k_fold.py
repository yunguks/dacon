import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.metrics import f1_score
from collections import OrderedDict
import timm
from sklearn.model_selection import StratifiedKFold
import argparse
import matplotlib.pyplot as plt

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
            self.backbone = timm.models.convnext.convnext_small(pretrained=True,num_classes=50)
        elif name =='eiffcient':
            self.backbone = timm.models.efficientnet.efficientnet_b0(pretrained=True,num_classes=50)
        else:
            self.backbone = timm.models.regnetx_064(pretrained=True,num_classes=50)
        # self.classifier = nn.Linear(1000, num_classes)
        # self.drop = nn.Dropout(0.5,inplace=True)
    def forward(self, x):
        x = self.backbone(x)
        # x = self.drop(x)
        # x = self.classifier(x)
        return x

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    classes,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = torch.sigmoid(inputs)
    # label smoothing
    targets = targets*(1-0.1)+0.1/50
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # 라벨마다 가중치
    max = classes[list(classes.keys())[0]][1]
    for i in range(targets.shape[0]):
        k = targets[i].argmax(0).item()
        more = torch.tensor(classes[list(classes.keys())[k]][1])
        loss[i] = loss[i]*max/more
        
    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

def make_class_info(df1,df2):
    classes = {}
    for i in df1['artist']:
        if i not in classes:
            # count 데이터 수
            classes[i] = 0
        else:
            classes[i] +=1
    # i번째 라벨 = [화가이름, image 수]
    convert_labels = sorted(classes.items(), key=lambda x : x[1], reverse=True)

    # key= 화가이름 value = [라벨번호, count 수]
    for i in range(len(convert_labels)):
        classes[convert_labels[i][0]]=[i,convert_labels[i][1]]

    # classes [label, count, years, genre, nationality]
    for name in classes.keys():
        for i in range(len(df2)):
            if df2.loc[i]['name'] == name:
                classes[name].extend(df2.loc[i].iloc[1:])
                classes[name] = tuple(classes[name])
    classes = OrderedDict(sorted(classes.items(), key = lambda t : t[1][1],reverse=True))

    return classes, convert_labels

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, class_info,transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms
        self.classes = class_info
    
    def __getitem__(self, index):
        img_path = './Dataset'+self.img_paths[index][1:]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.labels is not None:
            label = torch.zeros([50], dtype=torch.float32)
            label[self.classes[self.labels[index]][0]] = 1
            # print(f'artist name {self.labels[index]} , label = {self.classes[self.labels[index]][0]}')
            return image, label
        else:
            return image
    def __len__(self):
        return len(self.img_paths)      
    
    def getclasses(self):
        return self.classes

def competition_metric(true,pred):
    return f1_score(true,pred,average='macro')

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
            optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr_rate)
            # scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)
            scheduler = None
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
                },f'./checkpoint/best_{c}_regnet_{f1_score:.3f}.pth')
                print('Model Saved')
            else:
                early +=1
                if early > opt.stop:
                    print('Early stopping')
                    break
        total_history.append(history)
        c +=1
    return total_history

def cut_data(df,number,class_info,seed):
    print(f'total data : {len(df)}')
    for i in class_info:
        if class_info[i][1] > number:
            a = df[df['artist']==i]
            drop_index = list(a.sample(class_info[i][1]-number,random_state=seed)['id'])
            class_info[i][1]=number
            print(f'{i} delete {len(drop_index)}')
            df.drop(index=drop_index,inplace=True,axis=0)
    print(f'ater data : {len(df)}')
    return df 
    
def save_plot(history):
        for i in range(len(history)):
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            plt.plot(range(len(history[i]['train_loss'])),history[i]['train_loss'], label='train_loss')
            plt.plot(range(len(history[i]['train_loss'])),history[i]['val_loss'], label='val_loss')
            plt.title('Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(range(len(history[i]['f1_score'])), history[i]['f1_score'], label='f1_score')
            plt.title('F1_Score')
            plt.ylabel('f1_score')
            plt.xlabel('epoch')
            plt.legend()

            plt.savefig(f'./result/{i}st result image.png')

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
    