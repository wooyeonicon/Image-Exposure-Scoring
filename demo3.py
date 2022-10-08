# 作者 : 杨航
# 开发时间 : 2022/9/19 19:47

import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('F:/PythonRepository/picture/12.png')     # 平均值   5  6
                                                          # 中位数      6
                                                          # 直方图     6判断错误

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

# 处理图片
def due(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
    img_list = np.array(img)  # 转为numpy
    img_list_one = img_list.flatten() # 一维
    # img_bin = np.bincount(img_list_one)
    # img_bin = np.argmax(img_bin)    # 众数
    # sum = np.sum(img_list_one < img_average)
    # sum1 = np.bitwise_and(25 < img_list_one, img_list_one < 100).sum()

    circle_hist(img_list_one) # 直方图
    return img_list_one

# 方法一：平均值打分
def rate_ave(list):
    # 0-25,25-50,51-75,76-100,101-125,126-150,151-175,176-200,201-225,<256
    #  1     2      3    4       5        6      7        8      9     10
    img_average = np.average(list)  # 平均值
    rate = number(img_average)
    return rate

# 方法二：中位数打分
def rate_median(list):
    img_median = np.median(list)  # 中位数
    rate = number(img_median)
    return rate

# 方法三：直方图特征打分
def rate_hist(list):
    # 1.通过中位数的数量与最大值的数量的对比，如果差距不大，说明直方图相对较为准确
    # 获取直方图中每一块的数量，找到最大数量，再与中位数所在的块进行对比，数量差距不能过大
    median = np.median(list)  # 中位数
    max = 0  # 最大的数量
    flag,flag1 = 0,0
    for i in range(1,11):  # 像素数量最多的区域
        if i < 10:
            num = np.bitwise_and(25*(i-1) < list, list <= 25*i).sum()
        else:
            num = np.sum(list>23*(i-1))
        if max<num:
            max = num
            flag1 = i  # 需要知道数量最多的是哪个像素点区域

    for j in range(1,11):  # 中位数所在像素区域
        if median>25*(j-1) and median<=25*j:
            flag = j
    num_data = np.bitwise_and(25*(flag-1) < list, list <= 25*flag).sum() # 中位数所在像素区域的数量
    # print('median',median)
    # print('max',max)
    # print('num_data',num_data)
    # print('len',len(list))
    if ((max-num_data)/len(list))<0.25:  # 数据差距不大(=================0.25可以修改===================)
        rate_h = rate_median(median)   # 使用中位数打分
    else:                         # 第二步
        max_1 = flag1*25  # 最大数量
        max2,flag2 = 0,0
        for i in range(1,11):  # 找到第二大数量
            if i != flag1 and max2 < np.bitwise_and(25*(i-1) < list, list <= 25*i).sum():
                max2 = np.bitwise_and(25*(i-1) < list, list <= 25*i).sum()
                flag2 = i
        max_2 = flag2*25
        if (max_1<128 and max_2<128) or (max_1>128 and max_2>128):  # 两个最多的都在128像素左边
            rate_h = number((max_1+max_2)/2)
        else:                         # 在两边
            if max_1>225 or max_2>225:   # 说明图像有黑有白，但是白色居多，而且像素点很高（必定曝光）
                rate_h = 10
            elif max_2<25 or max_1<25:   # 说明图像有黑有白
                rate_h = 1
    return rate_h
def number(data): # 评分
    rate = 0
    if data<=25:
        rate = 1
    elif data<=50:
        rate = 2
    elif data<=75:
        rate = 3
    elif data<=100:
        rate = 4
    elif data<=125:
        rate = 5
    elif data<=150:
        rate = 6
    elif data<=175:
        rate = 7
    elif data<=200:
        rate = 8
    elif data<=225:
        rate = 9
    else:
        rate = 10
    return rate

# 三种评分做平均值
def ave(data1,data2,data3):
    return round((data1+data2+data3)/3,0)

if __name__ == '__main__':
    list = due(img)
    rate1 = rate_ave(list)
    rate2 = rate_median(list)
    rate3 = rate_hist(list)
    rate4 = ave(rate1,rate2,rate3)
    print('平均值打分',rate1)
    print('中位数打分',rate2)
    print('直方图打分',rate3)
    print('最终打分',rate4)



