import numpy as np

from readData import readGT, readResult
from makeInputData import make_input_data
from make_dataSet_for_model import make_data_loader_for_model
from model import get_Model, train, validatePerEpoch, test
from view_result import plotLoss

if __name__ == '__main__':

    print("reading ground truth data")
    gt = readGT("./data/ground_truth/groundTruth1.csv")

    print("reading result data")
    result = readResult("./data/detect/result1.csv")

    print("making input data")
    input, correct = make_input_data(result, gt)
    print("input data")
    print(input)
    print("correct data")
    print(correct)

    print("getting train, validation, test data ...")
    ratio = np.array([0.8, 0.1, 0.1])
    train_loader, validation_loader, test_loader = make_data_loader_for_model(input, correct, ratio)

    model, criterion, optimizer, device = get_Model()

    print('training model ... ')
    epochNum = 1000
    models, trainLoss = train(train_loader, model,  criterion, optimizer, device, epochNum)
    plotLoss(trainLoss)

    print("validating model ... ")
    val_loss_perEpoch, val_l2_perEpoch, l2_best_model = validatePerEpoch(validation_loader, models, criterion, device)
    # print("val length = ", len(val_loss_perEpoch))
    plotLoss(val_loss_perEpoch)
    plotLoss(val_l2_perEpoch)

    print("testing model ... ")
    test(test_loader, model, criterion, device)
