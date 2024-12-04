from readData import readGT, readResult



if __name__ == '__main__':

    print("reading ground truth data")
    gt = readGT("./data/ground_truth/groundTruth1.csv")
    print("reading result data")
    result = readResult("./data/detect/result1.csv")
