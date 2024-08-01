import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# 数据加载
Y = np.loadtxt('C:\\Users\sy\Desktop\LSTM_DCNN/CAN_X.txt', dtype=int)
X_test = Y[:, :-1]
Y_test = Y[:, -1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 将数据重塑为LSTM输入格式 (样本数, 时间步数, 特征数)
time_steps = 1  # 假设每个样本为一个时间步
num_features = X_test.shape[1]


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 参数设置
input_size = num_features
hidden_size = 128
num_layers = 5
num_classes = 5  # 假设5分类问题
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# 模型实例化
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import time
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('lstm_classifier.pth'))

model.eval()

print("Model loaded!")

time1 = time.time()
for i in range(X_test.shape[0]):
    c_X_test = X_test[i].reshape(-1, time_steps, num_features)
    c_X_test = torch.tensor(c_X_test, dtype=torch.float32)
    outputs = model(c_X_test)


time2 = time.time()
print(time2-time1)

















