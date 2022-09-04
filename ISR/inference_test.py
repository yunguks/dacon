import torch
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from test_train import SRCNN,CustomDataset
import zipfile
from PIL import Image

def inference(model, test_loader ,device):
    model.to(device)
    model.eval()
    pred_img_list = []
    name_list = []
    with torch.no_grad():
        for lr_img, file_name in tqdm(iter(test_loader)):
            lr_img = lr_img.float().to(device)

            pred_img = model(lr_img)

            for pred, name in zip(pred_img,file_name):
                pred = pred.cpu().clone().detach().numpy()
                pred = pred.transpose(1,2,0)
                pred = pred*255

                pred_img_list.append(pred.astype('uint8'))
                name_list.append(name)
    return pred_img_list,name_list

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print(device)
    print(f'torch version : {torch.__version__}')
    torch.cuda.empty_cache()
    #gc.collect()
    os.environ['CUDA_LAUNCH_BLOCKING']="1"

    csv_path = os.getcwd()+'/data/train.csv'
    df = pd.read_csv(csv_path)

    BATCH_SIZE =2
    IMG_SIZE=2048

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    test_path = os.getcwd()+'/data/test.csv'
    test_list = pd.read_csv(test_path)
    test_dataset = CustomDataset(test_list,test_transform,train_mode=False)
    test_loader = DataLoader(test_dataset,BATCH_SIZE,False)

    model = SRCNN()
    model = model.cpu()
    # map_location -- 이전 사용하던 gpu와 다른 모델일 경우
    checkpoint = torch.load('checkpoint/best_SRCNN2_0.01026.pth',map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    pred_img_list, pred_name_list = inference(model, test_loader,device)

    #os.chdir('~/dacon/ISR/data/submission/')
    os.chdir('./data/submission/')
    sub_imgs = []
    for path, pred_img in tqdm(zip(pred_name_list,pred_img_list)):
        pred_img = Image.fromarray(pred_img)
        pred_img.save(path)
        sub_imgs.append(path)

    submission = zipfile.ZipFile('../submission1.zip','w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()
    print('Done')
