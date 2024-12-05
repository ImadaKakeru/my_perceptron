import numpy as np

def make_trajectory(positions_in_image):
    # result = [[x_0, y_0], [x_1, y_1], ...]
    center = np.array([320, 240])

    diff = positions_in_image - center
    # print(diff)

    trajectory = []
    point = [0,0]
    for i in range(len(diff)):
        point[0] += diff[i][0]
        point[1] += diff[i][1]
        trajectory.append(point[:])

    trajectory = np.array(trajectory)
    return trajectory
