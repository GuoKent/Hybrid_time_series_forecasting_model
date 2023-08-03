import torch
from main_informer import *

test_file = 'FUDS_0C_data.csv'
model_para = '0C_DST_epochs=1_seq=96_lebel=48_pred=24'
model_name = 'checkpoint.pth'

# 模型路径
model_path = f'./checkpoints/{model_para}/{model_name}'

if __name__ == '__main__':
    for ii in range(args.itr):
        # setting record of experiments
        model = Exp(args)
        # para import
        model.Model_load(model_path)

        # FILE Change in here

        model.test(args)

    torch.cuda.empty_cache()
