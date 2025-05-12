import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import timm
import matplotlib.pyplot as plt
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from torch.multiprocessing import freeze_support

import warnings
warnings.filterwarnings(action='ignore') 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class PadSquare(ImageOnlyTransform):
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w, c = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)
        return image

    def get_transform_init_args_names(self):
        return ("border_mode", "value")
    

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)

    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['rock_type']),
                    y=train_df['rock_type']
                    )
    
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    
    best_score = 0
    best_model = None

    os.makedirs("./output", exist_ok=True)
    best_model_save_path = "./output/best_model.pth"

    plt.ion()
    print("Start training")
    x_arr=[]
    rec_loss = [[],[]]
    rec_acc=[[],[]]
    class_acc=[[],[],[],[],[],[],[]]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 4))
    
    start_time = time.time()

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_preds, true_train_labels = [], []
        train_loss = []
        model_save_path = f"./output/model_epoch{epoch}.pth"

        for imgs, labels in tqdm(iter(train_loader), desc=f"Epoch {epoch}"):
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            train_preds += output.argmax(dim=1).detach().cpu().numpy().tolist()
            true_train_labels += labels.detach().cpu().numpy().tolist()

        _train_loss = np.mean(train_loss)
        _train_score = f1_score(true_train_labels, train_preds, average='macro')

        _val_loss, _val_score,class_accuracy = validation(model, criterion, val_loader, device)

        print(f'Train Loss: {_train_loss:.5f}, Train Marco F1: {_train_score:.5f}')
        print(f'Val Loss: {_val_loss:.5f}, Val Macro F1: {_val_score:.5f}')
        torch.save(model.state_dict(), model_save_path)

        rec_loss[0].append(_train_loss)
        rec_loss[1].append(_val_loss)

        rec_acc[0].append(_train_score)
        rec_acc[1].append(_val_score)


        for i, acc in enumerate(class_accuracy):
            class_acc[i].append(acc)
            print(f"  Class {i+1}: {acc:.4f}")
        
        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {best_model_save_path}")


        to_numpy_loss = np.array(rec_loss)
        to_numpy_acc=np.array(rec_acc)
        x_arr = np.arange(len(rec_loss[0]))
        
        # 실시간 그래프 업데이트
        ax1.clear()
        # 손실 그래프
        ax1.plot(x_arr, to_numpy_loss[0], '-', label='Train val', marker='o')      
        ax1.plot(x_arr, to_numpy_loss[1], '--', label='Valid val', marker='o')
        ax1.legend(fontsize=15)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch', size=15)
        ax1.set_ylabel('Loss', size=15)
        
        ax2.clear()
        # 정확도 그래프
        ax2.plot(x_arr, to_numpy_acc[0], '-', label='Train acc', marker='o')
        ax2.plot(x_arr, to_numpy_acc[1], '--', label='Valid acc', marker='o')
        ax2.legend(fontsize=15)
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch', size=15)
        ax2.set_ylabel('acc', size=15)

        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()
    plt.tight_layout() 
    plt.savefig("graph.png")
    # plt.show()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f"Training time {total_time_str}")

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')

        cm = confusion_matrix(true_labels, preds)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return _val_loss, _val_score,class_accuracy


# 답안지 작성
def inference(model, test_loader, device):
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds




if __name__ == '__main__':
    # freeze_support()  # 윈도우에서 멀티프로세싱 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CFG = {
    'IMG_SIZE':224,
    'EPOCHS':50,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41
    }

    seed_everything(CFG['SEED']) # Seed 고정

    # # 이미지 경로 가져오기
    all_img_list = glob.glob('./train/*/*')

    df = pd.DataFrame(columns=['img_path', 'rock_type'])

    df['img_path'] = all_img_list

    # # rock_type을 상위 폴더명으로 지정
    df['rock_type'] = df['img_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    train_df, val_df, _, _ = train_test_split(df, df['rock_type'], test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])

    le = preprocessing.LabelEncoder()

    train_df['rock_type'] = le.fit_transform(train_df['rock_type'])
    val_df['rock_type'] = le.transform(val_df['rock_type'])

    train_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=(0.1, 0.1), scale_limit=(0.0, 0.3), rotate_limit=45, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CoarseDropout(max_holes=8, min_holes=2, p=0.2),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)
        ], p=0.4),
        A.VerticalFlip(p=0.2),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # 05-11 ImageNet의 평균 및 표준편차로 정규화
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # 05-11 ImageNet의 평균 및 표준편차로 정규화
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = CustomDataset(train_df['img_path'].values, train_df['rock_type'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=4,pin_memory=True,prefetch_factor=2)
   
    val_dataset = CustomDataset(val_df['img_path'].values, val_df['rock_type'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4,pin_memory=True,prefetch_factor=2)

    model = timm.create_model('legacy_xception', pretrained=True, num_classes=7)


    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"],weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)

    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

    test = pd.read_csv('./test.csv')
    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    preds = inference(infer_model, test_loader, device)

    submit = pd.read_csv('./sample_submission.csv')
    submit['rock_type'] = preds
    submit.to_csv('./baseline_submit.csv', index=False)