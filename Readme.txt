main_informer.py可改参数
train_data, train_data2 两个训练文件
test_data 在model_test_port里面修改，不用在这里改
train_mode: 用几个文件训练，1 or 2
epochs 每轮可训练多少个epoch
itr 训练几轮
patience 当误差连续patience次小于一定数值，触发早停机制
gpu 用第几块gpu训练
网络结构参数看着改，部分修改需要连着网络一起修改

model_test_port.py
test_file 测试文件名
model_path 模型路径


网络流程中都有注释，不懂再问