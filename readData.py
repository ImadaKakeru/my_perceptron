import pandas as pd
import numpy as np
def readGT(file_path):
    column = ['x', 'y']
    csv = pd.read_csv(file_path)
    # gt_x =[]
    gt_x = csv[column[0]].values
    gt_y = csv[column[1]].values

    gt_x = np.array(gt_x)
    gt_y = np.array(gt_y)
    # print(gt_x)
    # print(gt_y)
    gt = np.stack((gt_x, gt_y), axis=1)
    print(gt)


def readResult(file_path):
    csv = pd.read_csv(file_path)
    column = ['x', 'y']

    result_x = csv[column[0]].values
    result_y = csv[column[1]].values

    result_x = np.array(result_x)
    result_y = np.array(result_y)

    result = np.stack((result_x, result_y), axis=1)
    print(result)
