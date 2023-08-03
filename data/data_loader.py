import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


# 自定数据预处理类
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='DST_0C.csv',
                 target='SOC', scale=True, inverse=False, timeenc=0,
                 freq='s', cols=None, SOC=True, label='range', begin=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.SOC = SOC
        self.label = label
        self.begin = begin
        # 执行__read_data__()函数
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        temp = self.data_path.split('_')[-1]
        temp = temp.split('.')[0]
        path = self.root_path + temp + '/' + self.data_path
        df_raw = pd.read_csv(path)
        df_raw.insert(loc=0, column='date', value=0)
        # SOC起点
        df_raw = df_raw[int((1 - self.begin) * len(df_raw)):]
        if self.label == 'time':
            date = pd.date_range('2022-01-01', periods=len(df_raw), freq='1s')
            date = date.strftime('%X')
        else:
            date = range(len(df_raw))
        df_raw['date'] = date
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)  # 读取列名
            cols.remove('Profile')       # 去掉文件类型
            cols.remove(self.target)     # 去掉SOC
            cols.remove('date')          # 去掉date
        df_raw = df_raw[['date'] + cols + [self.target]]

        # MS:多特征预测单元素
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 除去date列，从1开始
            df_data = df_raw[cols_data]     # 读取数据
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data  # 训练集分离
            self.scaler.fit(train_data.values)             # 预处理，归一化
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]  # 提取时间
        if self.label == 'time':
            df_stamp['date'] = pd.to_datetime(df_stamp.date)  # 格式转化pandas
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        else:
            data_stamp = (df_stamp['date'] - np.mean(df_stamp['date'])) / np.std(df_stamp['date'])
            data_stamp = np.expand_dims(data_stamp, axis=1)

        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index                                   # index是随机索引，数据起始位置
        s_end = s_begin + self.seq_len                    # 一个seq_len长度，数据终止位置
        r_begin = s_end - self.label_len                  # decoder输入起始位置，重叠部分的开始
        r_end = r_begin + self.label_len + self.pred_len  # decoder输入的终止位置

        seq_x = self.data_x[s_begin:s_end]  # [seq_len, feature_size]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]  # [label_len+pred_len, feature_size] 即 [48+24, feature_size]
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 时间特征
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


