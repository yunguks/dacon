import pandas as pd
import torch
import argparse
import os
from utils import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
def get_args():
    parser = argparse.ArgumentParser(description="QAT training test")
    parser.add_argument("--device", type=int, default=0,help="select device number")
    parser.add_argument("--batch", type=int, default=8, help="batch parameter. Default 8")
    parser.add_argument("--length" , type=int, default=50)
    parser.add_argument("--txt", type=str, default=None, help="output metrix save in txt")
    return parser.parse_args()

def inference(test_loader, device):
    
    with torch.no_grad():
        model_name = "x3d_xs"
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)
        
        crash_model = model.copy()
        crash_model.blocks.append(nn.Linear(400,1))
        crash_model.blocks.append(nn.Sigmoid())
        crash_model.to(device)
        crash_model.eval()
        
        model.blocks.append(nn.Linear(400,12))
        model.blocks.append(nn.Softmax())
        model.to(device)
        model.eval()
        
        preds = []
        for videos in tqdm(iter(test_loader)):
            # crash 유무
            videos = videos.to(device)
            
            logit = crash_model(videos)
            
            logit = (logit > 0.5)
            # 부딪힌 경우  
            if logit:
                result =1
                logit = model(videos)
                result += logit.argmax(1).detach().cpu().numpy().tolist()
                
            else:
                result =0
            preds.append(result)
    return preds

if __name__=="__main__":
    args = get_args()
    print(args)
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')  
    
    df = pd.read_csv("./data/test.csv")
    
    test_dataset = CustomDataset(df['video_path'].values, None)
    test_loader = DataLoader(test_dataset, batch_size = args.batch, shuffle=False, num_workers=0)
    
    preds = inference(test_loader, device)
    submit = pd.read_csv('./sample_submission.csv')
    submit['label'] = preds
    submit.to_csv('./baseline_submit.csv', index=False)