import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import train_k_fold
from myutils import make_class_info,competition_metric,cut_data,CustomDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn.functional as F

def inference(k,model, test_loader, device,num_classes=50):
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
                out = F.softmax(out,dim=1)
                for b in range(len(out)):
                    model_pred.append(out[b].detach().cpu().numpy().tolist())
            total.append(model_pred)
    result = []
    for j in range(len(total[0])):
        out = np.zeros(num_classes)
        for i in range(k):
            out += np.array(total[i][j])
            if j ==1:
                print(total[i][j])
                print('----')
                print(out)
        if j==1:
            print(out.argmax(0))
        result.append(out.argmax(0))
    return result

if __name__=='__main__':
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    print(device)
    print(f'torch version : {torch.__version__}')

    df = pd.read_csv('./Dataset/train.csv')
    new_df = pd.read_csv('./Dataset/artists_info.csv')

    # class별 정보, i번째 라벨은 label_to_name[i][0]
    class_info , label_to_name = make_class_info(df,new_df)

    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229,0.224,0.224), max_pixel_value=255),
        ToTensorV2()
    ])
    test_df = pd.read_csv('./Dataset/test.csv')
    test_img_paths = test_df['img_path'].values
    test_dataset = CustomDataset(test_img_paths, None,class_info, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = train_k_fold.BaseModel()
    preds = inference(5,model, test_loader, device)
    print(preds[:5])
    result = []
    for i in preds:
        result.append(label_to_name[i][0])
    print(result[:5])
    submit = pd.read_csv('./Dataset/sample_submission.csv')
    submit['artist']=result
    submit.head()
    submit.to_csv('./Dataset/submit.csv', index=False)