import torch
import torchvision.datasets
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data


torch.manual_seed(1);

# 超参数区
EPOCH=1;
BATCH_SIZE=64;
TIM_STEP=28;      # rnn输入的图片高度或者时间步数（T=28）
INPUT_SIZE=28;    # rnn 每步输入值/图片每行像素（input_size=28）
LR=0.03;
DOWNLOAD_MNIST=True;  # 如果原始数据存在，会自动处理而不重新下载

# 数据集区
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=DOWNLOAD_MNIST)



train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]


# =========================
# ✅✅✅【必须手敲】模型结构 + forward（核心）
# 目的：理解结构、张量shape流、为什么这么取最后一步
# =========================
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM( # 定义RNN的LSTM的结构
            input_size=28, #输入层参数 :怎么得到的? 相片的像素(m*n /timestap)
            hidden_size=64, # 隐藏层的维度,手动定义,绝对模型的表达能力
            num_layers=1, # 定义LSTM RNN的层数
            batch_first=True, #输入输出shape约定 ? 这个不理解是什么
        )
        self.out=nn.Linear(64,10);# 定义输出层 输出层的输入,看上一层隐藏层的输出,输出层的输出,看具体的任务要求
    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None) #和我之前学的forward 流程不太一样
        out =self.out(r_out[:,-1,:])
        return out

# 实例化神经网络
rnn=RNN()

# 训练配置区
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

# 执行训练

for epoch  in range(EPOCH):
    for step,(x,b_y) in enumerate(train_loader):
        b_x=x.view(-1,28,28)
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每隔一定的步数打印训练过程和测试准确率
        if step % 50 == 0:  # 每50个batch打印一次
            # 计算测试准确率
            rnn.eval()  # 设置为评估模式
            with torch.no_grad():  # 测试时不需要计算梯度
                test_output = rnn(test_x.view(-1, 28, 28))
                pred_y = torch.max(test_output, 1)[1].data
                accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
            rnn.train()  # 恢复训练模式
            
            print('Epoch: %d | train loss: %.4f | test accuracy: %.2f' % (epoch, loss.item(), accuracy))





test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print('prediction number:',pred_y)
print('real number:',test_y[:10])


# # 【可复制脚手架】测试与输出
# # =========================
# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')