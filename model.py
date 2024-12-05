import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_Model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用するCPUの決定
    model = Net().to(device)

    #最適化手法の決定
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer, device

def train(train_loader, model, criterion, optimizer, device, epochNum):
    model.train()
    running_loss = 0.0
    models = []
    loss_list = []

    for i in tqdm(range(epochNum)):
        running_loss = 0
        for images, coordinate in train_loader:
            # print(images.shape)
            # ここで images のチャンネル次元を修正する
            # images = images.permute(1, 0, 2, 3)
            images, coordinate = images.to(device), coordinate.to(device)
            # print("images.shape ", images.shape)
            # print("coordinate.shape ", coordinate.shape)
            optimizer.zero_grad()
            outputs = model(images)
            # print("outputs", outputs)
            # print('coordinate', coordinate)
            loss = criterion(outputs, coordinate)
            # print('loss', loss)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        # print("running loss = ", running_loss)
        if i%5 == 0:
            # print("epochNum = ", i)
            models.append(copy.deepcopy(model))
        train_loss = running_loss / len(train_loader)
        # print("epochNum, train_loss = ", i, train_loss)
        loss_list.append(train_loss)

    # 学習済みのモデルを返す。
    return models, loss_list