import argparse
import os
import torch
import random

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='informer', help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, default='DST', help='data')
parser.add_argument('--SOC', type=bool, default=False, help='whether get SOC as feature,enc_in, dec_in=4 if True else 3')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')  # 数据根目录
parser.add_argument('--train_data', type=str, default='DST_30C.csv', help='data file')              # 训练数据文件名
parser.add_argument('--train_data1', type=str, default='DST_30C.csv', help='data file')             # 训练数据文件名
parser.add_argument('--train_data2', type=str, default='US06_30C.csv', help='data file')            # 训练数据文件名
parser.add_argument('--train_data3', type=str, default='FUDS_30C.csv', help='data file')            # 训练数据文件名
parser.add_argument('--test_data', type=str, default='FUDS_30C.csv', help='data file')              # 测试数据文件名
parser.add_argument('--train_mode', type=int, default=2, help='train 1 or 2 file at a time')        # 测试数据文件名
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='SOC', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--label', type=str, default='range', help='range or time')
parser.add_argument('--begin', type=float, default=1.0, help='begin SOC: range [0, 1]')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')  # 96个输入长度，48+48
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')   # decoder 48个SOC真值，也就是encoder后面48个
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')                # 24个预测长度
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')  # feature_size + SOC
parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')  # 输出维度,SOC,1
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')  # 走几遍encoder
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')  # 采样因子数
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')  # 训练几轮 即 itr*epochs
parser.add_argument('--epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)  #

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()
# args, unknown = parser.parse_known_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

# 定义模型
Exp = Exp_Informer

if __name__ == '__main__':
    random.seed(1)
    exp = Exp(args)  # set experiments
    # setting record of experiments
    # 模型文件命名用
    temp = args.train_data.split('_')[-1]  # 温度
    temp = temp.split('.')[0]
    if args.train_mode == 1:
        mode = args.train_data.split('_')[0]  # 数据模式
        setting = '{}_{}_epochs={}_seq={}_lebel={}_pred={}'.format(temp, mode, args.epochs, args.seq_len,
                                                                   args.label_len, args.pred_len)
    else:
        mode1 = args.train_data1.split('_')[0]  # 数据模式
        mode2 = args.train_data2.split('_')[0]  # 数据模式
        setting = '{}_{}+{}_epochs={}_seq={}_lebel={}_pred={}'.format(temp, mode1, mode2, args.epochs, args.seq_len,
                                                                   args.label_len, args.pred_len)

    for ii in range(args.itr):
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        args.train_data = args.train_data1
        exp.train(setting)

        # train another file
        if args.train_mode == 2:
            args.train_data = args.train_data2
            exp.train(setting)

        torch.cuda.empty_cache()

