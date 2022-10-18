import torch
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset

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
                classes[name] = classes[name]
    classes = OrderedDict(sorted(classes.items(), key = lambda t : t[1][1],reverse=True))

    return classes, convert_labels

def competition_metric(true,pred):
    return f1_score(true,pred,average='macro')

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

            plt.savefig(f'./result/{i+1}st result image.png')

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