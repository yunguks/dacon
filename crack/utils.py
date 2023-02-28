from torch.utils.data import Dataset
import torch
import cv2
import random
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list=None,args=None,transform=None):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.args = args
        self.transform = transform
        
    def __getitem__(self, index):
        frames = self.get_video("./data"+self.video_path_list[index][1:])
        if self.transform:
            frames = self.aug(self.transform, frames)
        frames = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
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
        imgs = {}
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frames):
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if i==0:
                imgs['image'] = img  
            else:
                imgs[f"image{i}"]=img
        return imgs
    
    def aug(self,transforms, images):
        res = transforms(**images)
        images = np.zeros((len(images),180, 320,3), dtype=np.uint8)
        images[0, :, :, :] = res["image"]
        for i in range(1, len(images)):
            images[i, :, :, :] = res[f"image{i}"]
        return images


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def collate_fn(batch):
    indice = torch.randperm(len(batch))
    value = round(np.random.beta(0.2,0.2),1)
    
    if len(batch[0])==2:
        img = []
        label = []
        for a,b in batch:
            img.append(a)
            label.append(torch.LongTensor(b))
        img = torch.stack(img)
        label = torch.stack(label)
        
        shuffle_label = label[indice]
        
        label = value * label + (1 - value) * shuffle_label
    else:
        img = torch.stack(batch)    
    shuffle_img = img[indice]
    
    img = value * img + (1 - value) * shuffle_img
    
    if len(batch[0])==2:
        return img, label
    else:
        return img