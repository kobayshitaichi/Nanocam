import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

import random

def make_filepath_list():
    train_file_list = []
    valid_file_list = []
    num_sumples = len(os.listdir('../dataset/train/images/off'))
    for top_dir in os.listdir('../dataset/train/images/'):
        file_dir = os.path.join('../dataset/train/images/',top_dir)
        if file_dir == '../dataset/train/images/._.DS_Store' or file_dir == '../dataset/train/images/.DS_Store':
            continue
        file_list = os.listdir(file_dir)
        random.shuffle(file_list)
        file_list = file_list[:num_sumples]

        

        #８割を学習データ、２割を検証データ
        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join('../dataset/train/images/',top_dir,file).replace('\\','/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join('../dataset/train/images/',top_dir,file).replace('\\','/') for file in file_list[num_split:]]

    return train_file_list, valid_file_list

train_file_list, valid_file_list = make_filepath_list()

if '../dataset/train/images/on/._.DS_Store' in train_file_list:
    train_file_list.remove('../dataset/train/images/on/._.DS_Store')
if '../dataset/train/images/on/._.DS_Store' in valid_file_list:
    valid_file_list.remove('../dataset/train/images/on/._.DS_Store')
if '../dataset/train/images/off/._.DS_Store' in train_file_list:
    train_file_list.remove('../dataset/train/images/off/._.DS_Store')
if '../dataset/train/images/off/._.DS_Store' in valid_file_list:
    valid_file_list.remove('../dataset/train/images/off/._.DS_Store')
if '../dataset/train/images/on/.DS_Store' in train_file_list:
    train_file_list.remove('../dataset/train/images/on/.DS_Store')
if '../dataset/train/images/on/.DS_Store' in valid_file_list:
    valid_file_list.remove('../dataset/train/images/on/.DS_Store')
if '../dataset/train/images/off/.DS_Store' in train_file_list:
    train_file_list.remove('../dataset/train/images/off/.DS_Store')
if '../dataset/train/images/off/.DS_Store' in valid_file_list:
    valid_file_list.remove('../dataset/train/images/off/.DS_Store')

print('学習データ数 : ', len(train_file_list))
print(train_file_list[:3])
print('検証データ数 : ', len(valid_file_list))
print(valid_file_list[:3])


class ImageTransform(object):
    def __init__(self,resize,mean,std):
        self.data_transform = {
            'train': transforms.Compose([ 
                #データオーグメンテーション
                transforms.RandomHorizontalFlip(),
                #画像をresizexresizeの大きさに統一する
                transforms.Resize((resize,resize)),
                #Tensor型に変換する
                transforms.ToTensor(),
                #色情報の標準化
                transforms.Normalize(mean,std)
            ]),
            'valid': transforms.Compose([
                transforms.Resize((resize,resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((resize,resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

resize = 300
mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)
transform = ImageTransform(resize,mean,std)


class SurgeryDataset(data.Dataset):
    def __init__(self,file_list,classes,transform=None,phase='test'):
        self.phase = phase
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase
    def __len__(self):
        #画像の枚数を返す
        return len(self.file_list)
        
    def __getitem__(self,index):
        #前処理した画像データのTensor形式のデータとラベルを取得

        #指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path)

        #画像の前処理を実施
        img_transformed = self.transform(img,self.phase)

        #画像ラベルをファイル名から抜き出す
        if self.phase == 'train' or self.phase=='valid':
            label = self.file_list[index].split('_')[-2]
        else:
            label = self.file_list[index].split('_')[-1][:-4]
        

        #ラベル名を数値に変換
        label = self.classes.index(label)

        return img_transformed, label

surgery_classes = ['on','off']

#Datasetの作成
train_dataset = SurgeryDataset(
    file_list=train_file_list,classes=surgery_classes,
    transform=ImageTransform(resize,mean,std),
    phase='train'
)
valid_dataset = SurgeryDataset(
    file_list=valid_file_list,classes=surgery_classes,
    transform=ImageTransform(resize,mean,std),
    phase='valid'
)

index = 0

#バッチサイズの指定
batch_size = 1

#DataLoaderを作成
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=1,shuffle=True
)

valid_dataloader = data.DataLoader(
    valid_dataset,batch_size=1,num_workers=1,shuffle=True
)

batch_iterator = iter(train_dataloader)
inputs, labels = next(batch_iterator)

class Net(pl.LightningModule):
    #ネットワークで使用する層を記述
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3,
        out_channels=64, kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64,
        out_channels=64, kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,
        stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64,
        out_channels=128, kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128,
        out_channels=128, kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
        stride=2)

        self.fc1 = nn.Linear(in_features=128 * 75 * 75, 
        out_features=128)
        self.fc2 = nn.Linear(in_features=128,
        out_features=5)
        self.fc3 = nn.Linear(in_features=5,
        out_features=2)       

    #順伝搬処理を記述
    def forward(self,x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1,128*75*75)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x,dim=1)

        return x
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)

        return {'loss':loss, 'y_hat':y_hat, 'y':y,
        'batch_loss':loss.item()*x.size(0)}

    #各エポック終了時の処理を記述
    def training_epoch_end(self, train_step_outputs):
        y_hat = torch.cat([val['y_hat'] for val in 
        train_step_outputs], dim=0)
        y = torch.cat([val['y'] for val in 
        train_step_outputs], dim=0)
        epoch_loss = sum([val['batch_loss'] for val in 
        train_step_outputs]) / y_hat.size(0)

        preds = torch.argmax(y_hat,dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', epoch_loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)

        print('-------- Current Epoch {} --------'.format(self.current_epoch + 1))
        print('train Loss: {:.4f} train Acc: {:.4f}'.format(epoch_loss, acc))
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        return {'y_hat': y_hat, 'y': y, 'batch_loss': loss.item() * x.size(0)}
    
    def validation_epoch_end(self, val_step_outputs):
        # x_hatを一つにまとめる
        y_hat = torch.cat([val['y_hat'] for val in val_step_outputs], dim=0)
        y = torch.cat([val['y'] for val in val_step_outputs], dim=0)
        epoch_loss = sum([val['batch_loss'] for val in val_step_outputs]) / y_hat.size(0)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', epoch_loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

        print('valid Loss: {:.4f} valid Acc: {:.4f}'.format(epoch_loss, acc))

    # 最適化手法を記述する
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        return optimizer



net = Net()

es = pl.callbacks.EarlyStopping(monitor='val_loss')

trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[es],
    gpus = 0,
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)

torch.save(net.state_dict(),'../models/model.pth')
