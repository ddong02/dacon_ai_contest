from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import VGG16, ResNet50V2, Xception, InceptionResNetV2, EfficientNetB0, EfficientNetB7, ConvNeXtBase
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import F1Score, CategoricalAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split

import sklearn
import albumentations as A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

### 학습 데이터프레임을 학습/검증 데이터프레임으로 분리하는 함수.
def Split_DataFrame(train_df, test_size=0.25, shuffle=True):

    ### 데이터프레임 자체에서 원-핫 인코딩 진행.
    train_df = pd.concat([train_df, pd.get_dummies(train_df['label'])], axis=1)
    train_df, val_df = train_test_split(train_df, test_size=test_size, stratify=train_df['label'], shuffle=shuffle)

    return train_df, val_df

### 학습/테스트 데이터가 들어있는 파일을 입력받아 DataFrame으로 반환해주는 함수.
def Create_DataFrame(train_dir, test_dir, test_csv, test_size=0.25, data_shuffle=True):
    train_image_filenames = []
    train_image_gubuns = []
    test_image_filenames = []

    ### 학습 이미지 절대 경로 구하는 로직
    for dirname, _, filenames in os.walk(train_dir):
        for filename in filenames:
            train_image_filenames.append(os.path.join(dirname, filename))

    ### 학습 이미지 label 구하는 로직
    for filename in train_image_filenames:
        slice_index1 = filename.find('/', 18)
        slice_index2 = filename.find('/', slice_index1+1)
        gubun = filename[slice_index1+1 : slice_index2]
        train_image_gubuns.append(gubun)

    ### 테스트 이미지 절대 경로 구하는 로직
    for dirname, _, filenames in os.walk(test_dir):
        for filename in filenames:
            test_image_filenames.append(os.path.join(dirname, filename))

    train_df = pd.DataFrame(train_image_filenames, columns=['img_path'])
    train_df['label'] = train_image_gubuns

    test_df = pd.read_csv(test_csv)
    test_image_filenames.sort()
    test_df['img_path'] = test_image_filenames

    train_df, val_df = Split_DataFrame(train_df, test_size=test_size, shuffle=data_shuffle)

    return train_df, val_df, test_df

TRAIN_PATH = '/content/open/train'
TEST_PATH = '/content/open/test'
TEST_CSV = '/content/open/test.csv'
test_size = 0.25
shuffle = True
train_df, val_df, test_df = Create_DataFrame(TRAIN_PATH, TEST_PATH, TEST_CSV, test_size, shuffle)

class Stone_Dataset(Sequence):
    def __init__(self, image_filenames, label_array, image_size = (224, 224), batch_size=64, augmentor=None, pre_func=None, shuffle=False):
        self.image_filenames = image_filenames
        self.label_array = label_array
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        self.shuffle = shuffle

        if self.shuffle:
            pass

    def __len__(self):
        return int(np.ceil(self.image_filenames.shape[0]/self.batch_size))

    def __getitem__(self, index):
        batch_image_filenames = self.image_filenames[index * self.batch_size :  (index+1) * self.batch_size]
        batch_image_array = np.zeros((batch_image_filenames.shape[0], self.image_size[0], self.image_size[1], 3))

        if self.label_array is not None:
            batch_label_array = self.label_array[index * self.batch_size : (index+1) * self.batch_size]

        for index in range(batch_image_filenames.shape[0]):
            image_array = cv2.cvtColor(cv2.imread(batch_image_filenames[index]), cv2.COLOR_BGR2RGB)

            if self.augmentor is not None:
                image_array = self.augmentor(image=image_array)['image']

            image_array = cv2.resize(image_array, (self.image_size[1], self.image_size[0]))

            if self.pre_func is not None:
                image_array = self.pre_func(image_array)

            batch_image_array[index] = image_array

        if self.label_array is not None:
            return batch_image_array, batch_label_array
        else:
            return batch_image_array

    def on_epoch_end(self):
        if self.shuffle:
            self.image_filenames, self.label_array = sklearn.utils.shuffle(self.image_filenames, self.label_array)
        else:
            pass

### Sequence_DataSet 만드는 함수.
def Create_Sequence_DataSet(train_df, val_df, test_df, image_size = (224, 224), batch_size=128, train_augmentor=None, check_augmentor=None, pre_func=None):
    ### 학습 이미지 파일경로
    train_image_filenames = train_df['img_path'].values
    train_label_array = train_df.iloc[:, 2:].values

    ### 검증 이미지 파일경로
    val_image_filenames = val_df['img_path'].values
    val_label_array = val_df.iloc[:, 2:].values

    ### 테스트 이미지 파일경로
    test_image_filenames = test_df['img_path'].values

    ### 학습/검증/테스트 Sequence DataSet 만들기.
    train_ds = Stone_Dataset(train_image_filenames, train_label_array, batch_size=batch_size, augmentor=train_augmentor, pre_func=pre_func)
    val_ds = Stone_Dataset(val_image_filenames, val_label_array, batch_size=batch_size, augmentor=check_augmentor, pre_func=pre_func)
    test_ds = Stone_Dataset(test_image_filenames, label_array=None, batch_size=batch_size, augmentor=check_augmentor, pre_func=pre_func)

    return train_ds, val_ds, test_ds

def create_model(model_name='Xception', weight='imagenet', image_size=(224, 224), fully_connected_layer=None, optimizer=None, metrics=None, loss='categorical_crossentropy', MODEL_DEBUG=True):
    input_tensor = Input(shape=(image_size[0], image_size[1], 3))

    if model_name == 'VGG16':
        pre_model = VGG16(include_top=False, input_tensor=input_tensor)
    elif model_name == 'ResNet50V2':
        pre_model = ResNet50V2(include_top=False, input_tensor=input_tensor)
    elif model_name == 'Xception':
        pre_model = Xception(include_top=False, input_tensor=input_tensor)
    elif model_name == 'ResNet50V2':
        pre_model = ResNet50V2(include_top=False, input_tensor=input_tensor)
    elif model_name == 'InceptionResNetV2':
        pre_model = InceptionResNetV2(include_top=False, input_tensor=input_tensor)
    elif model_name == 'EfficientNetB0':
        pre_model = EfficientNetB0(include_top=False, input_tensor=input_tensor)
    elif model_name == 'EfficientNetB7':
        pre_model = EfficientNetB7(include_top=False, input_tensor=input_tensor)
    elif model_name == 'ConvNeXtBase':
        pre_model = ConvNeXtBase(include_top=False, input_tensor=input_tensor)

    x = pre_model.output
    output = fully_connected_layer(x)

    model = Model(inputs=input_tensor, outputs=output)

    if MODEL_DEBUG==True:
        model.summary()

    model.compile(optimizer=optimizer, metrics=metrics, loss=loss)

    return model

# 모델 학습시키는 함수.
def Model_Train(model, train_ds, val_ds=None, epochs=10, callbacks=None):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model, history

def pre_func(image_array):
    return image_array/255.0

def fully_connected_layer_func(x):
    x = GlobalAveragePooling2D()(x)
    x = Dense(7, activation='softmax')(x)
    return x

def lr_fn1(epoch):
    LR_START = 1e-5
    LR_MAX = 1e-3
    LR_RAMPUP_EPOCHS = 2
    LR_SUSTAIN_EPOCHS = 1
    LR_STEP_DECAY = 0.75

    def calc_fn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = ((LR_MAX - LR_START) / LR_RAMPUP_EPOCHS) * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//2)

        print('epoch:', epoch, 'lr:', lr)

        return lr

    return calc_fn(epoch)

def lr_fn2(epoch):
    LR_START = 1e-6
    LR_MAX = 1e-4
    LR_RAMPUP_EPOCHS = 1
    LR_SUSTAIN_EPOCHS = 1
    LR_STEP_DECAY = 0.75

    def calc_fn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = ((LR_MAX - LR_START) / LR_RAMPUP_EPOCHS) * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//2)

        print('epoch:', epoch, 'lr:', lr)

        return lr
    return calc_fn(epoch)

augmentor_light1 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=(0.2, 0.1), scale_limit=(0.5, 1.0), rotate_limit=90, p=0.5)
], p=0.5)

augmentor_middle1 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=(0.2, 0.1), scale_limit=(0.7, 1.0), rotate_limit=90, p=0.5),
    A.Blur(blur_limit=8, p=0.4),
    A.CoarseDropout(num_holes_range=(10, 30), p=0.2),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1)
    ], p=0.4)
], p=0.5)

augmentor_heavy1 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=(0.2, 0.1), scale_limit=(0.7, 1.0), rotate_limit=90, p=0.5),
    A.Blur(blur_limit=10, p=0.5),
    A.CoarseDropout(num_holes_range=(10, 30), p=0.3),
    A.RandomRain(p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1)
    ], p=0.5)
], p=0.6)

lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='min', verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
save_cb = ModelCheckpoint(filepath='/content/gdrive/MyDrive/Colab Notebooks/z.대회 & 프로젝트/2025_건설용 자갈 암석 종류 분류 AI 경진대회_DACON/ConvNeXtBase{epoch:02d}-{val_loss:.2f}.weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
save_cb1 = ModelCheckpoint(filepath='/content/gdrive/MyDrive/Colab Notebooks/z.대회 & 프로젝트/2025_건설용 자갈 암석 종류 분류 AI 경진대회_DACON/Xception{epoch:02d}-{val_loss:.2f}.weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
save_cb2 = ModelCheckpoint(filepath='/content/gdrive/MyDrive/Colab Notebooks/z.대회 & 프로젝트/2025_건설용 자갈 암석 종류 분류 AI 경진대회_DACON/Xception{epoch:02d}-{val_loss:.2f}.weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
lr_scheduler1 = LearningRateScheduler(lr_fn1,verbose=1)
lr_scheduler2 = LearningRateScheduler(lr_fn2,verbose=1)

class Config:
    train_dir = '/content/open/train' # 학습 데이터가 들어있는 디렉토리
    test_dir = '/content/open/test' # 테스트 데이터가 들어있는 디렉토리
    test_csv_path = '/content/open/test.csv' # 테스트 csv파일
    test_size = 0.25 # Split_DataFrame()함수에서 사용하는 test_size
    data_shuffle = True # Split_DataFrame()함수에서 사용하는 shuffle
    image_size = (224, 224) # 데이터 변환시 적용할 이미지 사이즈(+ 모델 input_shape)
    batch_size = 128
    train_augmentor = augmentor_middle1 # 학습 데이터 augmentor
    check_augmentor = None # 검증/데스트 데이터 augmentor
    pre_func = pre_func # 학습 preprocessing function
    model_name = 'ConvNeXtBase'
    weight = 'imagenet'
    fully_connected_layer = fully_connected_layer_func # 모델 output에 적용할 fully-connected-layer funciton
    optimizer=Adam(0.0001)
    metrics=[F1Score('macro'), CategoricalAccuracy()]
    loss='categorical_crossentropy' # 모델 compile시 적용할 loss function
    MODEL_DEBUG = True # 모델 summary 보여주는 유무
    n_epochs=15 # 파인튜닝 안 할 때 epoch
    first_epochs = 10 # 파인튜닝 할 때 첫 번째 epoch
    second_epochs = 10 # 파인튜닝 할 때 두 번째 epoch
    first_train_layer_index = -2 # 파인튜닝 할 때 첫 번째 학습 시 freeze하는 layer 범위([:first_train_layer_index]로 들어감.)
    callbacks = [lr_cb, ely_cb, save_cb] # 파인튜닝 안 할 때 callback
    first_callbacks = [lr_cb, ely_cb, save_cb1] # 파인튜닝 할 때 첫 번째 callback
    second_callbacks = [lr_scheduler2, ely_cb, save_cb2] # 파인튜닝 할 때 두 번째 callback
    model_weight_file = '/content/gdrive/MyDrive/Colab Notebooks/z.대회 & 프로젝트/2025_건설용 자갈 암석 종류 분류 AI 경진대회_DACON/ConvNeXtBase06-0.40.weights.h5'

config = Config
def end_to_end(Fine_tuning = False, load_weight=False, config=config):
    train_df, val_df, test_df = Create_DataFrame(config.train_dir, config.test_dir, config.test_csv_path, test_size=config.test_size, data_shuffle=config.data_shuffle)

    train_ds, val_ds, test_ds = Create_Sequence_DataSet(train_df, val_df, test_df, image_size = config.image_size, batch_size=config.batch_size,
                                                        train_augmentor=config.train_augmentor, check_augmentor=config.check_augmentor, pre_func=config.pre_func)

    model = create_model(model_name=config.model_name, weight=config.weight, image_size=config.image_size, fully_connected_layer=config.fully_connected_layer,
                         optimizer=config.optimizer, metrics=config.metrics, loss=config.loss, MODEL_DEBUG=config.MODEL_DEBUG)

    if load_weight:
        print('########## MODEL LOAD ##########')
        print()
        model.load_weights(config.model_weight_file)

    # 파인튜닝 하고싶으면 True로 놓으면 됨.
    if Fine_tuning:
        history = []
        print('########## FINE TUNING 시작 ##########')
        print()

        for layer in model.layers[:config.first_train_layer_index]:
            layer.trainable = False

        print('########## FULLY_CONNECTED_LAYER 학습 시작 ##########')

        model, history1 = Model_Train(model, train_ds, val_ds=val_ds, epochs=config.first_epochs, callbacks=config.first_callbacks)
        history.append(history1)

        for layer in model.layers:
            if 'EfficientNet' in config.model_name:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
            else:
                layer.trainable = True

        print('########## FULL_LAYER 학습 시작 ##########')
        model, history2 = Model_Train(model, train_ds, val_ds=val_ds, epochs=config.second_epochs, callbacks=config.second_callbacks)
        history.append(history2)
    else:
        print('########## FULL_LAYER 학습 시작 ##########')
        model, history = Model_Train(model, train_ds, val_ds=val_ds, epochs=config.n_epochs, callbacks=config.callbacks)

    return model, history, test_ds

model, history, test_ds = end_to_end(Fine_tuning=False, load_weight=True, config=config)