import ffmpeg
import pandas as pd
import os
import shutil
import argparse
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-data',help='train or test')
parser.add_argument('-date')

args = parser.parse_args()
data = args.data
date = args.date

def split_test_data(date):
    #アノテーション結果をDataframe型にする
    path = '../videos/'+date+'/'+date+'.txt'
    df = pd.read_table(path,header=None)
    df.columns=columns=['a','b','c','start','e','end','g','length','onoff']
    df = df.drop(['a','b','c','e','g','end'],axis=1)
    

    #手術動画を分割
    print('{}: Split videos start'.format(date))
    path = '../videos/'+date+'/'+date+'.mp4'
    for i in tqdm(range(len(df))):
        s = str(i)
        stream = ffmpeg.input(path,ss=df['start'][i],t=df['length'][i])
        stream = ffmpeg.output(stream,'../videos/'+date+'/split_videos/videos_{}.mp4'.format(s.zfill(3)))
        ffmpeg.run(stream)

    #動画をフレーム毎の静止画に分割
    print('{}: Convert to Images start'.format(date))
    for i in tqdm(range(len(df))):
        s = str(i)
        path = '../videos/'+date+'/split_videos/videos_'
        path = path+s.zfill(3)+'.mp4'
        stream = ffmpeg.input(path)
        stream = ffmpeg.output(stream,'../dataset/test/'+date+'/images/'+s.zfill(5)+'_%7d_'+df['onoff'][i]+'.jpg',r=30)
        ffmpeg.run(stream)

    

    #分割動画を削除
    #shutil.rmtree('./TestImages/videos')

def split_train_data(date):
    #アノテーション結果をDataframe型にする
    path = '../videos/'+date+'/'+date+'.txt'
    df = pd.read_table(path,header=None)
    df.columns=columns=['a','b','c','start','e','end','g','length','onoff']
    df = df.drop(['a','b','c','e','g','end'],axis=1)
    

    #手術動画を分割
    if len(os.listdir('../videos/'+date+'/split_videos'))==0:
        print('{}: Split videos start'.format(date))
        path = '../videos/'+date+'/'+date+'.mp4'
        for i in tqdm(range(len(df))):
            s = str(i)
            stream = ffmpeg.input(path,ss=df['start'][i],t=df['length'][i])
            stream = ffmpeg.output(stream,'../videos/'+date+'/split_videos/videos_{}.mp4'.format(s.zfill(3)))
            ffmpeg.run(stream)

    #動画をフレーム毎の静止画に分割
    print('{}: Convert to images start'.format(date))
    for i in tqdm(range(len(df))):
        s = str(i)
        path = '../videos/'+date+'/split_videos/videos_'
        path = path+s.zfill(3)+'.mp4'
        stream = ffmpeg.input(path)
        stream = ffmpeg.output(stream,'../dataset/train/images/'+ df['onoff'][i] +'/'+ date+'_'+s.zfill(3)+'_'+df['onoff'][i]+'_%10d.jpg',r=5)
        ffmpeg.run(stream)
if data == 'test':
    split_test_data(date)
elif data == 'train':
    split_train_data(date)
    # list = ['20201217','20201229','20210525','20210601','20210604','20210914','20211116','20220215']
    # for i in list:
    #     split_train_data(i)
else:
    print('-data train or test')