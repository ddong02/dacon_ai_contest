# best_model 불러와서 추론과정 진행하는 코드

import torch
import pandas as pd
import timm
import albumentations as A
import cv2

from tqdm.auto import tqdm
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


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
    

def inference(model, test_loader, device):
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

    CFG = {
        'IMG_SIZE':224,
        'EPOCHS':50,
        'LEARNING_RATE':3e-4,
        'BATCH_SIZE':64,
        'SEED':41
    }

    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    le = preprocessing.LabelEncoder()

    best_model_save_path = "./output/best_model.pth"

    model = timm.create_model('legacy_xception', pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(best_model_save_path, map_location=device))
    model.eval()

    test = pd.read_csv('./test.csv')
    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    preds = inference(model, test_loader, device)

    submit = pd.read_csv('./sample_submission.csv')
    submit['rock_type'] = preds
    submit.to_csv('./baseline_submit.csv', index=False)