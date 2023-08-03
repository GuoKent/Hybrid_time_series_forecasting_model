import torch
from main_informer import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

test_file = 'FUDS_30C.csv'
model_para = '30C_DST+US06_epochs=20_seq=96_lebel=48_pred=24'
model_name = 'checkpoint.pth'
begin_SOC = 0.8
args.begin = begin_SOC

args.test_data = test_file

# 模型路径
model_path = f'./checkpoints/{model_para}/{model_name}'

if __name__ == '__main__':
    # setting record of experiments
    model = Exp(args)
    # para import
    model.Model_load(model_path)

    # FILE Change in here

    model.test(args, test_file)

    torch.cuda.empty_cache()
