import torch


def devide_dataSet(result, correct, ratio):
    train = []
    validation = []
    test = []
    return train, validation, test

def make_data_loader(result_data, correct_data):

    return


def make_data_loader_for_model(result, correct, ratio):
    train, validation, test = devide_dataSet(result=result, correct=correct, ratio=ratio)

    train = make_data_loader()
