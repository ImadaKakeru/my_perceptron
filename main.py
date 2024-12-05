from readData import readGT, readResult
from makeInputData import make_input_data


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


