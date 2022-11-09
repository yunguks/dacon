import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models

from sklearn.metrics import f1_score

GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
os.environ['CUDA_LAUNCH_BLOCKING']="1"
print(device)
print(f'torch version : {torch.__version__}')
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':8,
    'SEED':41
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
df = pd.read_csv('./Dataset/train.csv')

from collections import OrderedDict
classes = {}
for i in df['artist']:
    if i not in classes:
        # count 데이터 수
        classes[i] = 0
    else:
        classes[i] +=1
convert_labels = sorted(classes.items(), key=lambda x : x[1], reverse=True)


for i in range(len(convert_labels)):
    classes[convert_labels[i][0]]=[i,convert_labels[i][1]]

new_df = pd.read_csv('./Dataset/artists_info.csv',sep=',')

# classes [label, count, years, genre, nationality]
for name in classes.keys():
    for i in range(len(new_df)):
        if new_df.loc[i]['name'] == name:
            classes[name].extend(new_df.loc[i].iloc[1:])
            classes[name] = classes[name]
classes = OrderedDict(sorted(classes.items(), key = lambda t : t[1][1],reverse=True))

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

df = cut_data(df,200,classes,41)

train_df,val_df,_, _ = train_test_split(df, df['artist'].values, test_size=0.2, shuffle=True,random_state=CFG['SEED'])

def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values

train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)

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
            # labels df['artist']
            label[self.classes[self.labels[index]][0]] = 1
            # print(f'artist name {self.labels[index]} , label = {self.classes[self.labels[index]][0]}')
            return image, label
        else:
            return image
    def __len__(self):
        return len(self.img_paths)      
    
    def getclasses(self):
        return self.classes

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224,224),
    A.HorizontalFlip( p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, p=0.5),
    # A.CoarseDropout(max_holes=4, max_height=16, max_width=16, 
    #                          min_holes=None, min_height=16, min_width=16,always_apply=False, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229,0.224,0.224), max_pixel_value=255),
    #A.Normalize(max_pixel_value=255),
    # (HxWxC) -> (CxHxW)
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    #A.Normalize(max_pixel_value=255),
    A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229,0.224,0.224), max_pixel_value=255),
    ToTensorV2()
])

train_dataset = CustomDataset(train_img_paths, train_labels,classes, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True)

val_dataset = CustomDataset(val_img_paths,val_labels,classes, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

img_mean=(0.485, 0.456, 0.406)
img_std= (0.229,0.224,0.224)

from torchvision.models import convnext_large,ConvNeXt_Large_Weights
from torchvision import models

import timm
class BaseModel(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(BaseModel, self).__init__()
        #self.backbone = convnext_large(weight=ConvNeXt_Large_Weights.DEFAULT)
        # self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        # self.backbone = timm.models.regnetx_064(pretrained=True,num_classes=50)
        #self.backbone = timm.create_model('coatnet_3_224',pretrained=True)
        #self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        #self.backbone= models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=50)
        self.backbone = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        self.classifier = nn.Linear(1000, num_classes)
        self.drop = nn.Dropout(0.5,inplace=True)
    def forward(self, x):
        x = self.backbone(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x
import torchsummary
model = BaseModel()
torchsummary.summary(model, (3,224,224),device='cpu')

#pytorch 참고
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
    classes=classes
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
    # # 라벨마다 가중치
    # max = classes[list(classes.keys())[0]][1]
    # for i in range(targets.shape[0]):
    #     k = targets[i].argmax(0).item()
    #     more = torch.tensor(classes[list(classes.keys())[k]][1])
    #     loss[i] = loss[i]*max/more
        
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

from typing import Optional

def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Function that measures the Binary Cross Entropy between the target and input
    probabilities.
    See :class:`~torch.nn.BCELoss` for details.

    Examples::
        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> loss = F.binary_cross_entropy(torch.sigmoid(input), target)
        >>> loss.backward()
    """
    p = torch.sigmoid(inputs)
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            binary_cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if target.size() != input.size():
        raise ValueError(
            "Using a target size ({}) that is different to the input size ({}) is deprecated. "
            "Please ensure they have the same size.".format(target.size(), input.size())
        )

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    loss = torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
    # 라벨마다 가중치
    max = classes[list(classes.keys())[0]][1]
    for i in range(targets.shape[0]):
        k = targets[i].argmax(0).item()
        more = torch.tensor(classes[list(classes.keys())[k]][1])
        loss[i] = loss[i]*max/more

    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)

def train(model, optimizer, trian_loader, test_loader, scheduler, device,name='best'):
    model.to(device)
    history = {'train_loss':[],'val_loss':[],'f1_score':[]}
    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = sigmoid_focal_loss
    best_f1 = 0
    best_loss = 100
    count = 0
    for epoch in range(1,CFG['EPOCHS']):
        model.train()
        train_loss = []
        for img , label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.to(device)
            optimizer.zero_grad()

            model_pred = model(img)

            loss = criterion(model_pred, label,alpha=0.1)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            #print(loss.item())
            #break
        tr_loss  = np.mean(np.array(train_loss))

        val_loss, val_score = validation(model,criterion, test_loader, device)

        print(f'Epoch [{epoch}], Train Loss : {tr_loss:.5f}, Val Loss : {val_loss:.5f}, Val F1 Score : {val_score:.5f}')

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['f1_score'].append(val_score)

        if scheduler is not None:
            scheduler.step()
        if best_f1 < val_score:
            best_f1 = val_score
            count=0
            if val_score > 0.7:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },f'./checkpoint/best_{name}_{val_score:.3f}.pth')
                print('Model Saved')
        else:
            if count >50:
                print('early stopping')
                break
            count +=1
    return history

def competition_metric(true,pred):
    return f1_score(true,pred,average='macro')

def validation(model, criterion,test_loader, device):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label,alpha=0.1)

            val_loss.append(loss.item())
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.argmax(1).detach().cpu().numpy().tolist()
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1

# optimizer = torch.optim.Adam(params=model.parameters(), lr = 6e-5)
lr = 1e-2
optimizer = torch.optim.SGD(params=[
    {'params':model.backbone.parameters(), 'lr':0.1*lr},
    {'params':model.classifier.parameters(), 'lr':lr}
    ], lr=1e-2,momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=50,eta_min=1e-5)
# scheduler = None

history = train(model, optimizer, train_loader, val_loader, scheduler, device,name='test')
