import os
import argparse
import random

from tqdm import tqdm
from helper import VideoHelper
from utils import get_all_frames, select_speaking_person

def speaker_detect(video_file, args):
    '''
    根据视频，找出一段时间内的说话人。
    :param video_file: 视频文件地址
    :param args: 各类参数
    :return:
    '''
    if not os.path.exists(video_file):
        print("Video File not exist!")
        return -1

    video_helper = VideoHelper(predictor_path=args.predictor_path,
                               model_path=args.model_path,
                               output_dir=args.output_dir)

    frame_list = get_all_frames(video_file)
    start_idx, end_idx = args.start_time * args.fps, args.end_time * args.fps
    frame_list = frame_list[start_idx: end_idx]

    speaking_label_list = video_helper.face_recognition(frame_list,
                                                        using_cuda=args.using_GPU,
                                                        batch_size=args.batch_size,
                                                        model=args.dlib_model)

    mouth_data = {} # {"person_index": {"abs": [float, ], "rel": [float, ]}}
    print("Start Speaker Detection")
    for t in tqdm(range(len(speaking_label_list))):
        speaking_label = speaking_label_list[t]

        # 对于未出现的人的label，填充一个非零的较小值
        for key in mouth_data.keys():
            mouth_data[key]["abs"].append(0.0001 * (random.random() - 0.5))
            mouth_data[key]["rel"].append(0.0001 * (random.random() - 0.5))

        for info in speaking_label:
            idx, abs, rel = info["index"], info["abs_label"], info["rel_label"]
            if idx in mouth_data:
                mouth_data[idx]["abs"][-1] = abs
                mouth_data[idx]["rel"][-1] = rel
            else:
                mouth_data[idx] = {"abs": [abs], "rel": [rel]}

    select_idx, result = select_speaking_person(mouth_data)
    print("说话人下标：%d" % select_idx)

    return select_idx



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # 图像识别模型的参数
    parser.add_argument('--predictor_path', type=str, default="../model/shape_predictor_68_face_landmarks.dat",
                        help="parameters for dlib predictor")
    parser.add_argument('--model_path', type=str, default="../model/dlib_face_recognition_resnet_model_v1.dat",
                        help="parameters for dlib face recognition model")
    parser.add_argument('--dlib_model', type=str, default="cnn",
                        help="which type of model dlib uses for face recognition, cnn (for gpu)/ hog (for cpu)")
    parser.add_argument('--using_GPU', type=bool, default=True,
                        help="whether cuda is available")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size of face recognition when cuda is available")

    # 视频文件相关参数
    parser.add_argument('-v', '--input_video_path', type=str, required=True,
                        help='input video file')
    parser.add_argument('-s', '--start_time', type=int, default=0,
                        help='the start time of video (second)')
    parser.add_argument('-e', '--end_time', type=int, required=True,
                        help='the end time of video (second)')
    parser.add_argument('-f', '--fps', type=int, default=25,
                        help='the frame per second of video')

    # 输出路径参数
    parser.add_argument('-o', '--output_dir', type=str, default="../output",
                        help='directory of output, including face, audio and record')

    args = parser.parse_args()

    speaker_detect(video_file=args.input_video_path, args=args)
