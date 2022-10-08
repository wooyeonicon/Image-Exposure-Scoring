# 作者 : 杨航
# 开发时间 : 2022/9/24 22:43
import cv2
import numpy as np
def state(image_path):
    img = cv2.imread(image_path, 1)  # 读取彩色图
    img = cv2.resize(img, (500, 500))  # 设置大小
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # hsv转换
    thre = 0.175   # 设置阈值（阈值可调整）
    imageOverExposeBlockNum = 0 # 图像过曝光的块数
    imageBlocks = 0   # 一共经历的块数
    imageDarkBlockNum = 0   # 图像欠曝光的块数
    imgVChannels = img[:,:,2]  # 把所有V通道全部取出来
    step = 8  # 设置小区域为8*8
    i = 0
    while i < imgVChannels.shape[0]:  #行
        j=0
        while j < imgVChannels.shape[1]:  #列
            imageBlock = imgVChannels[i:i + step, j:j + step]  # 小区域
            mean = np.mean(imageBlock)  # 求小矩形的均值
            if mean > 233.0:   # 过曝光阈值
                imageOverExposeBlockNum += 1  # 过曝光度
            elif mean < 53.0:  # 欠曝光阈值
                imageDarkBlockNum += 1    # 欠曝光度
            imageBlocks += 1   # 记录小区域的数量
            j += step   # 列往后走8位
        i += step # 行往下走8位
    if imageDarkBlockNum / imageBlocks > thre:  # 欠曝光
        print('欠曝光')
    elif imageOverExposeBlockNum / imageBlocks > thre:  # 过曝光
        print('过曝光')
    else:   # 正常
        print('正常')
    overExpose = imageOverExposeBlockNum / imageBlocks * 100
    darkExpose = imageDarkBlockNum / imageBlocks * 100
    state = np.absolute(overExpose - darkExpose)/100
    return 1-state,overExpose,darkExpose
if __name__ == '__main__':
    print(state('F:/PythonRepository/picture/12.png'))


