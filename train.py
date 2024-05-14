# train_public 唯一字段
## known_outstanding_loan：借款人档案中未结信用额度的数量
## known_dero：贬损公共记录的数量
## app_type：是否个人申请

# train_internet 唯一字段
## sub_class：网络贷款等级之子级
## work_type：工作类型
# house_loan_status：房屋贷款状况（无房贷、正在还房贷、已经还完房贷）
## marriage：婚姻状态（未婚、已婚、离异、丧偶）
## offsprings：子女状态(无子女、学前、小学、中学、大学、工作)
## f5：匿名特征f0-f5，为一些网络贷款人行为计数特征的处理，不可合并！

# 删除f，
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # 从 LSTM 输出的最后一层中得到 LSTM 输出
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])  # 只选择 LSTM 输出的最后一个时间步
        return out

def init():
    x = pd.read_csv("./preprocessed_train_public.csv").drop(columns="loan_id").drop(columns="user_id").drop(columns="isDefault")
    input_dim = x.shape[1]  # 特征数量
    hidden_dim = 100  # 隐藏层维度
    layer_dim = 1  # LSTM 层的数量
    output_dim = 1  # 输出维度
    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, criterion, optimizer

model, criterion, optimizer = init()


def train(model, criterion, optimizer):
    x = pd.read_csv("./preprocessed_train_public.csv").drop(columns="loan_id").drop(columns="user_id")
    y = x["isDefault"]
    x = x.drop(columns="isDefault")
    
    x_train_tensor = torch.tensor(x.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    

def eval(model, criterion, optimizer):
    model.eval()  # 切换到评估模式
    loan_id = pd.read_csv("./test_public.csv")["loan_id"]
    x_test = pd.read_csv("./preprocessed_test_public.csv")
    x_test = x_test.drop(columns="loan_id").drop(columns="user_id")
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    test_dataset = TensorDataset(x_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    result_pd = pd.DataFrame(columns=["id", "isDefault"])
    results = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].unsqueeze(1)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # 将输出转换为二进制预测
            results.extend(predicted.numpy().flatten())
    for i in range(len(results)):
        results[i] = int(results[i])
        result_pd = pd.concat([result_pd, pd.DataFrame([{"id": loan_id[i], "isDefault": results[i]}])], ignore_index=True)
    result_pd.to_csv("submission.csv", index=False)

train(model, criterion, optimizer)
eval(model, criterion, optimizer)