import cv2
import os
import face_recognition
import dlib
import numpy as np

from tqdm import tqdm
from imutils import face_utils
from scipy.spatial import distance as dist

print("DLIB Use GPU:", dlib.DLIB_USE_CUDA)

class VideoHelper():

    def __init__(self, predictor_path, model_path, output_dir):

        self.predictor = dlib.shape_predictor(predictor_path)
        self.embedding = dlib.face_recognition_model_v1(model_path)
        self.font = cv2.FONT_ITALIC

        self.face_dataset = []          # elememt {"index":index, "embedding":128D-vector}
        self.face_data_dir = os.path.join(output_dir, "face")
        if not os.path.exists(self.face_data_dir):
            os.mkdir(self.face_data_dir)
        else:
            for file in os.listdir(self.face_data_dir):
                os.remove(os.path.join(self.face_data_dir, file))

    def face_recognition(self, image_list, using_cuda=True, batch_size=16, model="cnn"):
        '''
        :param image_list: 待处理的图像列表。
        :param using_cuda: 是否使用GPU加速，如果使用GPU，则推荐使用batch cnn模型
        :param batch_size: 一个batch的大小
        :param model: cnn/hog，hog模型在cpu的环境下运行较快
        :return:
        '''
        location_list = []
        print("Start Face Detection")
        if using_cuda and model=="cnn":
            for i in tqdm(range(0, len(image_list), batch_size)):
                batch_location = face_recognition.batch_face_locations(image_list[i: i+batch_size],
                                                                       number_of_times_to_upsample=0,
                                                                       batch_size=batch_size)
                location_list = location_list + batch_location
        else:
            for i in tqdm(range(len(image_list))):
                image = image_list[i]
                location = face_recognition.face_locations(image, model=model)
                location_list.append(location)

        #  face_image, loc = self.face_extract(image, loc)
        assert len(image_list)==len(location_list)

        speaking_label_list = []
        for i in range(len(location_list)):
            image, locations = image_list[i], location_list[i]
            speaking_label = []
            for loc in locations:
                face_image, dlib_loc = self.face_extract(image, loc)
                check_result = self.face_check(image, dlib_loc)
                if not check_result["flag"]:
                    self.face_save(check_result["embedding"], face_image, check_result["index"])
                abs_label, rel_label = self.speaking_label(check_result["shape"])
                speaking_label.append({"index": check_result["index"], "abs_label": abs_label, "rel_label": rel_label})
            speaking_label_list.append(speaking_label)

        return speaking_label_list

    def face_extract(self, image, location):
        '''
        :param image: 待处理的一帧图像
        :param location: 人脸的坐标
        :return: face_image：提取的人脸图像, face_loc：人脸在原图上的坐标
        '''
        # 注意两种location格式不同，最后统一为：dlib.rectangle
        start_width, start_height, width, height = location[3], location[0], location[1] - location[3], location[2] - location[0]
        face_image = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                face_image[i][j] = image[start_height + i][start_width + j]

        # 重写人脸坐标的类
        face_loc = dlib.rectangle(start_width, start_height, start_width+width, start_height+height)
        return face_image, face_loc

    def face_check(self, image, face_loc):
        '''
        :param image: 未经裁剪的一帧画面图片
        :param face_image: 截取的人脸图片（彩色）
        :param face_loc: 人脸在画面中的位置
        :return: {"flag": 是否为认识的人脸, "index": 下标, "embedding": 128维向量, "shape": 人脸关键点坐标}
        '''

        # 计算128维人脸特征向量的欧氏距离
        def return_euclidean_distance(feature_1, feature_2):
            feature_1 = np.array(feature_1)
            feature_2 = np.array(feature_2)
            dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
            return dist

        face_embedding, shape = self._get_face_feature(image, face_loc)

        # 寻找欧氏距离最小的脸
        distance_list = [return_euclidean_distance(face_embedding, face["embedding"]) for face in self.face_dataset]
        min_idx = distance_list.index(min(distance_list)) if distance_list else -1

        # 根据欧氏距离判断
        if min_idx>=0 and distance_list[min_idx] < 0.4:
            result = {"flag": True, "index": min_idx, "embedding": face_embedding, "shape": shape}
        else:
            result = {"flag": False, "index": len(self.face_dataset), "embedding": face_embedding, "shape": shape}

        return result

    def face_save(self, face_embedding, face_image, index):
        '''
        :param face_embedding: 128维脸部的特征向量
        :param face_image: 脸部图片
        :return:
        '''
        # 计入dataset
        person_info = {"index": len(self.face_dataset), "embedding": face_embedding}
        self.face_dataset.append(person_info)

        # 输出到文件
        output_file = os.path.join(self.face_data_dir, "Person_%d.jpg" % (index))
        cv2.imwrite(output_file, face_image)

        return

    def speaking_label(self, shape):
        '''
        :param shape: 人脸关键点坐标
        :param history_label: 唇动指标的历史记录
        :return: abs_label：绝对指标, rel_label:相对指标
        '''

        # 计算方法：计算关键点62-68，63-67，64-66之间平均距离，作为绝对指标
        # 计算dist(62,68)/dist(51,59), dist(63,67)/dist(52,58), dist(64,66)/dist(53,57)的平均值作为相对指标
        a = dist.euclidean(shape[61], shape[67])
        A =  a / dist.euclidean(shape[50], shape[58])
        b = dist.euclidean(shape[62], shape[66])
        B = b / dist.euclidean(shape[51], shape[57])
        c = dist.euclidean(shape[63], shape[65])
        C = c / dist.euclidean(shape[52], shape[56])
        abs_label = float((a + b + c) / 3)
        rel_label = float((A + B + C) / 3)

        return abs_label, rel_label

    def _get_face_shape(self, image, face_loc):
        '''
        :param image: 一帧图像画面（不需要专门切出脸部）
        :param face_loc: 脸部的位置
        :return: shape:人脸关键点坐标
        '''
        shape = self.predictor(image, face_loc)
        return face_utils.shape_to_np(shape)

    def _get_face_feature(self, image, face_loc):
        '''
        :param image: 一帧图像画面（不需要专门切出脸部）
        :param face_loc: 脸部的位置
        :return: face_embedding:人脸的128维特征, shape:人脸关键点坐标
        '''
        # 获取人脸关键点特征，参数（图像，内部形状预测的边界）
        shape = self.predictor(image, face_loc)

        # 获取人脸的128维特征向量，参数（图像，关键点信息）
        face_embedding = self.embedding.compute_face_descriptor(image, shape)

        # 将shape转为numpy
        shape = face_utils.shape_to_np(shape)
        return face_embedding, shape

    def clean_memory(self):
        '''
        清楚dlib运行时候所占据的显存
        :return: None
        '''
        del face_recognition.api.cnn_face_detector

