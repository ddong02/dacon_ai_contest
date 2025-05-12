device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':50,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41
}

seed_everything(CFG['SEED']) # Seed 고정

# 이미지 경로 가져오기
all_img_list = glob.glob('./train/*/*')

df = pd.DataFrame(columns=['img_path', 'rock_type'])
df['img_path'] = all_img_list
# rock_type을 상위 폴더명으로 지정
df['rock_type'] = df['img_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

df


train_df, val_df, _, _ = train_test_split(df, df['rock_type'], test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])
le = preprocessing.LabelEncoder()
train_df['rock_type'] = le.fit_transform(train_df['rock_type'])
val_df['rock_type'] = le.transform(val_df['rock_type'])

train_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

test_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_dataset = CustomDataset(train_df['img_path'].values, train_df['rock_type'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=4,pin_memory=True,prefetch_factor=2)

val_dataset = CustomDataset(val_df['img_path'].values, val_df['rock_type'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4,pin_memory=True,prefetch_factor=2)

model = timm.create_model('resnet50', pretrained=True, num_classes=7)
model.eval()
# opotimizer 수정
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"],weight_decay=1e-4)
# dh fix
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)

test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)