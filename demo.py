# 作者 : 杨航
# 开发时间 : 2022/9/18 19:14

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
img1 = cv2.imread('F:/PythonRepository/picture/5.png')
img2 = cv2.imread('F:/PythonRepository/picture/5.png')
img3 = cv2.imread('F:/PythonRepository/picture/6.png')

# 图片展示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destoryAllWindows()
# 直方图
def circle_hist(data):
    plt.figure()
    plt.hist(data)
    plt.show()
# 折线图
def circle2(x,y):
    plt.figure()
    plt.plot(x,y)
    plt.show()

# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# cv_show('img',img2)

# # 计算灰度图的均值和方差
# img2_list = np.array(img2)
# ave = np.average(img2_list) # 均值
# img_one_list = img2_list.reshape(-1)
# # 偏离128的均值
# sum = 0
# for i in range(len(img_one_list)):
#     sum += math.pow(img_one_list[i]-ave,2)
# variance = sum/len(img_one_list)
# print('均值，方差',ave,variance)
def one_gray(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = np.array(img1)
    ave = np.average(img1)
    print('灰度图',ave)

def three_kernel(img):
    # 三通道 GBR
    # 设置阈值 （欠曝光<100,正常 100-150,过曝光 > 150）暂时
    b, g, r = cv2.split(img)
    # 去噪（把一些异常值去掉）
    y = b.reshape(-1)  # y轴
    x = len(y)  # x轴
    x = np.arange(x)
    # plt.figure()
    # plt.plot(x, y)
    # plt.show()

    b = np.array(b)
    g = np.array(g)
    r = np.array(r)
    # 计算整张图片三通道的平均值：代表图片最终的大小
    ave = (np.average(b) + np.average(g) + np.average(r)) / 3
    print('彩色图',ave)
def bgr_hsv(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    print('v',v) # 明亮度
    y = v.reshape(-1)
    result1 = np.median(y)
    result2 = np.average(y)
    result3 = np.bincount(y)
    result3 = np.argmax(result3)
    result4 = y.var()
    print('中位数 平均数 众数 方差',result1,result2,result3,result4)
    x = len(y)
    x = np.arange(x)
    circle_hist(y)  # 直方图
    #circle2(x,y)
    #cv_show('hsv',hsv)
if __name__ == '__main__':
    one_gray(img1)
    three_kernel(img2)
    bgr_hsv(img3)


