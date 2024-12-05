import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
from view_result import viewResult

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 48)
        self.fc3 = nn.Linear(48, 96)
        self.fc4 = nn.Linear(96, 64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # print("x", x)
        return x

def get_Model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用するCPUの決定
    model = Net().to(device)

    #最適化手法の決定
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

def validatePerEpoch(val_loader, models, criterion, device):

    loss_per_epoch = []
    l2_error_per_epoch = []
    print("models length = ", len(models))
    for model in models:
        model.eval()

    with torch.no_grad():
        for model in models:
            individual_error = []
            running_loss = 0.0
            for images, coordinate in val_loader:
                images, coordinate = images.to(device), coordinate.to(device)
                outputs = model(images)
                loss = criterion(outputs, coordinate)
                running_loss += loss.item()
                print("outputs = ", outputs)
                # outputs *= 4
                # coordinate *= 4
                l2_distance = torch.norm(outputs - coordinate, 2, 1)
                individual_error.extend(l2_distance.cpu().numpy())  # リストに展開して追加


            val_loss = running_loss / len(val_loader)
            # print("valLoss = ", val_loss)
            loss_per_epoch.append(val_loss)

            l2_distance_tensor = 0.44 * torch.tensor(individual_error)
            mean_l2_distance = torch.mean(l2_distance_tensor, dim=0)
            std_l2_distance = torch.std(l2_distance_tensor, dim=0)
            l2_error_per_epoch.append(mean_l2_distance)

    print("loss per epoch = ", loss_per_epoch)
    print("l2 error per epoch = ", l2_error_per_epoch)
    # min_loss_index = loss_per_epoch.index(min(loss_per_epoch))
    min_loss_index_l2 = l2_error_per_epoch.index(min(l2_error_per_epoch))
    # print("min loss index = ", min_loss_index)
    print("min l2 loss = ", l2_error_per_epoch[min_loss_index_l2])
    print("min l2 loss index = ", min_loss_index_l2)

    # loss_best_model = models[min_loss_index]
    l2_best_model = models[min_loss_index_l2]
    # return loss_per_epoch, loss_best_model, l2_error_per_epoch, l2_best_model
    return loss_per_epoch, l2_error_per_epoch, l2_best_model


def test(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    individual_error = []
    result = []
    inputImg = []
    with torch.no_grad():
        for images, correct in tqdm(test_loader):
            images, correct = images.to(device), correct.to(device)
            outputs = model(images)

            result.extend(outputs.cpu().numpy())

            l2_distance = torch.norm(outputs - correct, 2, 1)
            # print("l2_distance = ", l2_distance)
            # バッチ単位ではなく、個々のデータのL2距離をリストに追加
            individual_error.extend(l2_distance.cpu().numpy())  # リストに展開して追加
            loss = criterion(outputs, correct)
            running_loss += loss.item()

    # 全てのL2距離をTensorに変換して平均と標準偏差を計算
    # print("individual error length = ", len(individual_error))
    l2_distance_tensor = 0.44 * torch.tensor(individual_error)
    # print("l2_distance_tensor = ", l2_distance_tensor)
    mean_l2_distance = torch.mean(l2_distance_tensor, dim=0)
    std_l2_distance = torch.std(l2_distance_tensor, dim=0)

    print(f"mean ± std = {mean_l2_distance} ± {std_l2_distance}")
    return