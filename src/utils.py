import os
import cv2
import xlwt
import random


def get_all_frames(video_file):
    capture = cv2.VideoCapture(video_file)
    result = []
    while True:
        ret, image = capture.read()  # 读取一帧画面
        if not ret:
            break
        result.append(image)
    return result


def count_open_close_sum(data_list):
    '''
    :param data_list: 一段时间内的唇动指标数据
    :return: res：一个浮点说，表示这段时间的唇动值，唇动值越大，说明说话的可能性越大
    '''
    # 做一阶差分，然后将连续的正数和负数堆叠，依次计算乘积，求和。为了检测连续、大幅的张合运动
    # 例如差分后：[1, 2, 4, -1, 2, -2, 1, -4, -6, 2, 7] -> [1+2+4, -1, 2, -2, 1, -4-6, 2+7]
    d_data = [data_list[i+1] - data_list[i] for i in range(len(data_list)-1)]
    d_data = [n if n != 0 else 0.0001 * (random.random()-0.5) for n in d_data]

    pole_up = []
    s = 0
    for num in d_data:
        if s*num >=0:
            s += num
        else:
            pole_up.append(s)
            s = num
    pole_up.append(s)

    res = 0
    for i in range(len(pole_up)-1):
        res += (-pole_up[i] * pole_up[i+1])
    res += (sum(data_list) / len(data_list))

    return res


def select_speaking_person(mouth_data):
    '''
    :param mouth_data: 根据每个人的唇动指标，确定这一段时间内的说话人。
    :return: select_idx:选中的说话人下标, result：每个人的唇动值
    '''
    # mouth_data = {"person_index": {"abs": [float, ], "rel": [float, ]}}
    result = {}

    max_rel, select_idx = -1, -1
    for key, value in mouth_data.items():
        result[key] = {"abs": count_open_close_sum(value["abs"]), "rel": count_open_close_sum(value["rel"])}
        if result[key]["rel"] > max_rel:
            max_rel, select_idx = result[key]["rel"], key

    return select_idx, result

