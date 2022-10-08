# 作者 : 杨航
# 开发时间 : 2022/9/24 22:04
import cv2
import os
import numpy as np
import math

def overExposeDetect(img_path,size):
    img = cv2.imread(img_path, 1) # 彩色图
    img = cv2.resize(img, size)   # 设置大小
    thre = 0.175   # 阈值
    status = "normal"  # 字符串描述
    flag = False    # 是否曝光
    if img.shape[2] != 1:
        hsvSpaceImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # hsv转换
    else:
        hsvSpaceImage = img.clone()
    hsvImageVChannels = hsvSpaceImage[:, :, 2]  # 把所有V通道全部取出来
    step = 8   #以8*8小窗口遍历V通道图像
    imageOverExposeBlockNum = 0
    imageBlocks = 0
    imageDarkBlockNum = 0
    #遍历
    i = 0
    while i < hsvImageVChannels.shape[0]:
        j = 0
        while j < hsvImageVChannels.shape[1]:
            imageBlock = hsvImageVChannels[i:i+step, j:j+step]
            mea = np.mean(imageBlock)# 求小矩形的均值
            if mea > 233.0:
                imageOverExposeBlockNum += 1  # 过曝光度
            elif mea < 53.0:
                imageDarkBlockNum += 1    # 欠曝光度
            imageBlocks += 1
            j += step
        i += step
    if imageDarkBlockNum/imageBlocks > thre:
        status = "dark"
        flag = True
    if imageOverExposeBlockNum/imageBlocks > thre:
        status = "overexposure"
        flag = True
    if flag == True:
        print(str(img_path) + "该图曝光度:" + str(imageOverExposeBlockNum/imageBlocks * 100) + " status:" + status)
        print(" 正常区间:(0," + str(thre*100) + "]")
    return flag
if __name__ == '__main__':
    print(overExposeDetect('F:/PythonRepository/picture/14.png',(500,500)))