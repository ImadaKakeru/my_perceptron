import numpy as np
import copy

def calc_speed(before, after, dt):
    # print("before: ", before)
    # print("before[0]",before[0])
    # print("after: ", after)
    diff = after - before
    speed = diff/dt
    return speed

def calc_acceleration(first, second, third, dt):
    speed_t1 = calc_speed(first, second, dt)
    speed_t2 = calc_speed(second, third, dt)
    accel_x = speed_t2[0] - speed_t1[0]
    accel_y = speed_t2[1] - speed_t1[1]
    accel = np.array([accel_x, accel_y])
    return accel

def calc_input_feature(first, second, third, dt):

    position = copy.deepcopy(third)
    speed = calc_speed(first, second, dt)
    acceleration = calc_acceleration(first, second, dt, dt)

    return np.array([position, speed, acceleration])

def make_input_data(positions, ground_truth):
    input_data = []
    correct_data = []
    dt = 0.1

    # print("positions size = ", len(positions))
    # print("ground_truth size = ", len(ground_truth))

    for i in range(len(positions) - 4 + 1):
        first = copy.deepcopy(positions[i])
        second = copy.deepcopy(positions[i+1])
        third = copy.deepcopy(positions[i+2])

        # print("first: ", first)
        # print("second: ", second)
        # print("third: ", third)

        input_data.append(calc_input_feature(first, second, third, dt))
        correct_data.append(ground_truth[i+3])

    input_data = np.array(input_data)
    flattened_input = [part.flatten() for part in input_data]
    correct_data = np.array(correct_data)

    return flattened_input, correct_data