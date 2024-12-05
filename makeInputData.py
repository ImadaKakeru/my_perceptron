import numpy as np
import copy

def calc_speed(before, after, dt):
    diff_x = after[0] - before[0]
    diff_y = after[1] - before[1]
    v_x = diff_x/dt
    v_y = diff_y/dt
    speed = np.array([v_x, v_y])
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

