import numpy as np

from readData import readGT, readResult
from makeInputData import make_input_data
from make_dataSet_for_model import make_data_loader_for_model

if __name__ == '__main__':

    print("reading ground truth data")
    gt = readGT("./data/ground_truth/groundTruth1.csv")

    print("reading result data")
    result = readResult("./data/detect/result1.csv")

    print("making input data")
    print("result", result)
    input, correct = make_input_data(result, gt)
    print("input data")
    print(input)
    print("correct data")
    print(correct)

    print("getting train, validation, test data ...")
    ratio = np.array([0.8, 0.1, 0.1])
    train, validation, test = make_data_loader_for_model(input, correct, ratio)

    print("train data")
    print(train)

    print("validation data")
    print(validation)

    print("test data")
    print(test)