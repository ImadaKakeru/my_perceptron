import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

def devide_data(trainRatio, valRatio, testRatio, result, correct):
    if len(result) != len(correct):
        print('Images, GT, data length is different')

    print(len(result))
    # 103個の、10データ1まとまり
    devideNum = int(len(result) / 10)
    print("devide Num = ", devideNum)
    trainNum = int(round(devideNum * trainRatio))*10
    print("train num = ", trainNum)
    valNum = int(round(devideNum * valRatio))*10
    print("val num = ", valNum)
    testNum = int(round(devideNum * testRatio))*10
    print("test num = ", testNum)

    train = result[:trainNum]
    trainGT = correct[:trainNum]
    val = result[trainNum:trainNum + valNum]
    valGT = correct[trainNum:trainNum + valNum]
    test = result[trainNum + valNum:]
    testGT = correct[trainNum + valNum:]

    return train, trainGT, val, valGT, test, testGT


def make_data_loader(result_data, correct_data, batch_size, shuffle=True):
    result_tensor = torch.FloatTensor(result_data)
    correct_tensor = torch.FloatTensor(correct_data)
    dataSet = TensorDataset(result_tensor, correct_tensor)

    # print("DataSet", dataSet)
    return DataLoader(dataSet, batch_size=batch_size, shuffle=shuffle)


def make_data_loader_for_model(result, correct, ratio):
    # ratio = [train_ratio, validation_ratio, test_ratio]
    train, trainGT, validation, validationGT, test, testGT = devide_data(trainRatio=ratio[0], valRatio=ratio[1], testRatio=ratio[2], result=result, correct=correct)
    train = make_data_loader(train, trainGT, 10)
    validation = make_data_loader(validation, validationGT, 10)
    test = make_data_loader(test, testGT, 10)

    return train, validation, test
