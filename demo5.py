# 作者 : 杨航
# 开发时间 : 2022/9/23 17:28
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('F:/PythonRepository/picture/7.png')

# 图片展示
def cv_show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    # cv2.destoryAllWindows()
# 直方图
def circle_hist(data):
    plt.figure()
    plt.hist(data,256)
    plt.show()

def due4(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 灰度图
    # img1 = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1 = clahe.apply(img)  # 2
    img_list = np.array(img).flatten()
    print('像素点总数量：',len(img_list))
    img1_list = np.array(img1).flatten()
    # img_list_u = np.unique(img_list)
    # img1_list_u = np.unique(img1_list)
    # print('两者长度',len(img_list_u),len(img1_list_u))
    # plt.plot(img_list_u,img1_list_u[0:len(img_list_u)])

    value1,value2 = due5(img_list) # 原图的两个value值
    value1_1,value1_2 = due5(img1_list)  # 均衡化之后的两个value值
    state1 = state(value1,value2)
    state2 = state(value1_1,value1_2)
    circle_hist(img_list)
    circle_hist(img1_list)
    res = np.hstack((img, img1))
    cv_show(res)

# 按区域划分
def due5(img_list):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
    # img_list = np.array(img).flatten()  # 转为numpy
    sum = len(img_list)
    sum_d = sum/3.0
    sum_i = sum_d*2
    sum_t = sum_d*3
    img_list1 = np.sort(img_list)[int(sum_d)]
    img_list2 = np.sort(img_list)[int(sum_i)]
    print(img_list1,img_list2)
    return img_list1,img_list2
def state(value1,value2):
    # 满分6分  0-5(0,1,2,3,4,5)
    num = 256/3
    if np.absolute(num-value1)<=num:
        state1 = np.absolute(num-value1)/num
    elif np.absolute(num-value1)>num and np.absolute(num-value1)<(2*num):
        state1 = np.absolute(num-value1)/(2*num)
    if np.absolute(2*num-value2)<=num:
        state2 = np.absolute(2*num-value2)/num
    elif np.absolute(2*num-value2)>num and np.absolute(2*num-value2)<(2*num):
        state2 = np.absolute(2*num-value2)/(2*num)
    state = state1+state2
    print('每一区域的误差：',state1,state2)
    print('评分：',4-state)
    return state

if __name__ == '__main__':
    due4(img)