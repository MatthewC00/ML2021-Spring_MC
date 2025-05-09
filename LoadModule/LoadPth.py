import torch
import torch.nn as nn

# 假设这是你保存模型时使用的模型结构
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义模型层
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = MyModel()

# 加载模型参数
model.load_state_dict(torch.load('model.pth'))

# 确保模型处于评估模式
model.eval()
