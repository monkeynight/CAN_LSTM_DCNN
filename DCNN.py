import numpy as np
import torch
import torch.nn as nn

# 数据合并

# 特征和标签分离
Y = np.loadtxt('C:\\Users\sy\Desktop\LSTM_DCNN/CAN_X.txt', dtype=int)
X_test = Y[:, :-1]
Y_test = Y[:, -1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
import time
class DCNN(nn.Module):
    def __init__(self, num_classes):
        super(DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64 * 19 * 8, 512)  # 根据新的输入形状调整线性层
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 参数设置
num_classes = 5  # 假设5分类问题
num_epochs = 10
batch_size = 16  # 调整批量大小以适应GPU内存
learning_rate = 0.001

# 模型实例化并移动到GPU
model = DCNN(num_classes)
model.load_state_dict(torch.load('dcnn_classifier.pth'))
model.eval()
print("Model loaded!")


time1 = time.time()
for i in range(X_test.shape[0]):
    c_X_test = X_test[i].reshape(-1, 1, 76, 35)
    c_X_test = torch.tensor(c_X_test, dtype=torch.float32)
    outputs = model(c_X_test)


time2 = time.time()
print(time2-time1)
# 数据加载器


