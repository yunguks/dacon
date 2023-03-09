import pandas as pd
import torch
import argparse
import os
from utils import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np 
from collections import Counter

import albumentations as A
import copy
def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    parser.add_argument("--device", type=int, default=0,help="select device number")
    parser.add_argument("--length" , type=int, default=50)
    parser.add_argument("--txt", type=str, default=None, help="output metrix save in txt")
    parser.add_argument("--batch", type=int, default=1, help="batch parameter")
    parser.add_argument("--img-size", nargs="+",type=int, default=[180,320], help="set input image size. Default 180+320")
    
    return parser.parse_args()

def inference(test_loader, device, args):
    
    with torch.no_grad():
        model_name = "x3d_s"
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
        
        pt_list = glob.glob("./checkpoint/*")
        
        crash_list=[]
        weather_list=[]
        time_list=[]
        for path in pt_list:
            if ".pt" in path:
                if "crash" in path:
                    crash_list.append(path)
                if "weather" in path:
                    weather_list.append(path)
                if "time" in path:
                    time_list.append(path)
        
        crash_model = copy.deepcopy(model)
        crash_model.blocks.append(nn.Dropout(0.3))
        crash_model.blocks.append(nn.Linear(400,3))
        crash_model.to(device)
        crash_model.eval()
        
        weather_model = copy.deepcopy(crash_model)
        weather_model.to(device)
        weather_model.eval()
        
        time_model = copy.deepcopy(model)
        time_model.blocks.append(nn.Linear(400,1))
        time_model.blocks.append(nn.Sigmoid())
        time_model.to(device)
        time_model.eval()
        
        preds = []
        for videos in tqdm(iter(test_loader)):
            # crash 유무
            videos = videos.to(device)
            # vote = torch.zeros(size=(args.batch,3))
            vote = [0,0,0]
            for p in crash_list:
                crash_model.load_state_dict(torch.load(p)['model_state_dict'])
                logit = crash_model(videos)
                logit = F.softmax(logit,dim=1)
                v = logit.squeeze(0).argmax(0).detach().cpu().numpy()
                vote[v]+=1
            c = np.argmax(vote)
            
            # 부딪힌 경우  
            if c != 0:
                result =6*(c-1) +1
                # 날씨
                vote= [0,0,0]
                for p in weather_list:
                    weather_model.load_state_dict(torch.load(p)['model_state_dict'])
                    logit = weather_model(videos)
                    v = logit.squeeze(0).argmax(0).detach().cpu().numpy()
                    vote[v]+=1
                w = np.argmax(vote)
                result += 2*(w)
                
                # 시간
                vote = 0
                for p in time_list:
                    time_model.load_state_dict(torch.load(p)['model_state_dict'])
                    logit = time_model(videos)
                    v = logit[0][0].detach().cpu().numpy() >= 0.5
                    vote += 1*v
                
                result += 1*(vote//3)
                
            else:
                result =0
            preds.append(result)
    return preds

if __name__=="__main__":
    args = get_args()
    print(args)
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')  
    
    df = pd.read_csv("./data/test.csv")
    
    val_transforms = A.Compose([
            A.Resize(height=args.img_size[0], width=args.img_size[1]),
            A.Normalize(max_pixel_value=255.0)
        ],additional_targets={f"image{i}":"image" for i in range(1, 50)})
    
    test_dataset = CustomDataset(df['video_path'].values, transform=val_transforms, args=args)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    preds = inference(test_loader, device, args)
    print(Counter(preds))
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['label'] = preds
    submit.to_csv('./data/baseline_submit.csv', index=False)