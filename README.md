## 项目介绍
本项目是基于视频的说话人识别。在给定一段视频的情况下，判定这段视频中的说话人是谁。目前只能够判断单一说话人的视频。如果需要对于多人轮流说话的视频判断每段时间的说话人，可以通过其他方式判定可能的说话人切换点，随后对每小段视频进行判断。

## 原理说明
本项目主要依据人脸关键点的变动。人脸关键点序号如图：
![人脸关键点](https://github.com/flandy2010/video-based-speaker-detection/blob/master/IMG/key%20point.jpg)

其中，62和68，63和67，64和66号点的距离可以衡量开口的大小。因此取三组点距离的平均值作为绝对开口值，并计算dist(62,68)/dist(51,59),dist(63,67)/dist(52,58),dist(64,66)/dist(53,57)的平均值作为相对开口值。

一个正在说话的人，开口值会不断从小到大，从大到小的变化。因此对于每个人的开口值做一阶差分。将相邻的正数和负数分别相加，最后得到一个正负相间的序列。随后将相邻的两个正负数的乘积的绝对值求和，作为最终的分数。分数越大的人，说话的可能性越高。

## 依赖环境安装
`pip install -r requirments`

## 程序运行
程序需要指定视频文件路径，起止时间。

```
python speaker-detect.py -v "video_file_path" -s start_time -e end_time
```

如果是没有GPU的情况下，请使用hog模型保证程序运行的效率：

```
python speaker-detect.py -v "video_file_path" -s start_time -e end_time --using_GPU False --dlib_model "hog"
```

而在有GPU的情况下（默认有GPU），推荐使用cnn模型，并搭配batch使用。可以通过--batch_size来设定batch的大小。

```
python speaker-detect.py -v "video_file_path" -s start_time -e end_time --batch_size 32
```

