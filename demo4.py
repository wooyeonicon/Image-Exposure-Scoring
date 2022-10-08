# 作者 : 杨航
# 开发时间 : 2022/9/21 22:22
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('F:/PythonRepository/picture/4.png')

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
def due2(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_3 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img1 = cv2.equalizeHist(img)  # 1
    img1_3 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img2 = clahe.apply(img)  #  2
    img2_3 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)


    cv2.imshow('img',img_3)
    cv2.imshow('img1',img1_3)
    cv2.imshow('img2',img2_3)
    cv2.waitKey(0)
    #cv2.destoryAllWindows()

    # img_list = np.array(img).flatten()
    # img1_list = np.array(img1).flatten()
    # img2_list = np.array(img2).flatten()
    #
    # circle_hist(img_list)
    # circle_hist(img1_list)
    # circle_hist(img2_list)
    #
    # res = np.hstack((img1, img2))
    # cv2.imshow('dst', res)
    # cv2.waitKey(0)

# 处理图片
def due(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
    #img = cv2.resize(img,(900,900))
    img_list = np.array(img)  # 转为numpy
    img_list_one = img_list.flatten() # 一维
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    img_list1 = np.array(cl1)
    img_list_one1 = img_list1.flatten() # 一维
    sum1,sum2 = 0,0
    for i in range(256):
        sum1 += img_list_one[i]
        sum2 += img_list_one1[i]
    print('总体差值',sum1-sum2)


    loss = []  # 差值数组
    sum_loss = 0  # 差值和
    for i in range(256):
        loss.append(abs(img_list_one[i] - img_list_one1[i]))
        sum_loss += loss[i]
    print('长度',len(img_list_one))
    print('变化后长度',len(img_list_one1))
    print('差值数组',loss)
    print('差值总和',sum_loss)
    circle_hist(img_list_one) # 直方图
    circle_hist(img_list_one1)
    cv_show(cl1)
    return img_list_one

def due3(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    v = np.array(v).flatten() # 一维
    circle_hist(v)
def due4(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 灰度图
    # img1 = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1 = clahe.apply(img)  # 2

    img_list = np.array(img).flatten()
    print('像素点总数量：',len(img_list))
    img1_list = np.array(img1).flatten()
    due5(img_list)
    due5(img1_list)
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
# 计算图像对比度
def due6(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img1 = np.array(img).ndim  # 查看维度
    img1 = np.array(img)
    x = np.array(img).shape[0]  # (800,1200)  # 使用4*4计算
    y = np.array(img).shape[1]
    # sum1 = img1.sum()
    sum = 0
    for i in range(x):      # x-1
        for j in range(y):  # y-1
            if i==0 and j==0: # 左上
                #value = np.power(img1[i+1][j]-img1[i][j],2)+np.power(img1[i][j+1]-img1[i][j],2)
                value = abs(img1[i+1][j]-img1[i][j])**2+abs(img1[i][j+1]-img1[i][j])**2
                sum += value
            elif i==0 and j!=0 and j!=y-1: # 上中
                #value = np.power(img1[i][j-1]-img1[i][j],2)+np.power(img1[i][j+1]-img1[i][j],2)+np.power(img1[i+1][j]-img1[i][j],2)
                value = abs(img1[i][j-1]-img1[i][j])**2+abs(img1[i][j+1]-img1[i][j])**2+abs(img1[i+1][j]-img1[i][j])**2
                sum += value
            elif i==0 and j==y-1:  # 右上
                #value = np.power(img1[i][j-1]-img1[i][j],2)+np.power(img1[i+1][j]-img1[i][j],2)
                value = abs(img1[i][j-1]-img1[i][j])**2+abs(img1[i+1][j]-img1[i][j])**2
                sum += value
            elif i!=0 and i!=x-1 and j==0:  # 左中
                #value = np.power(img1[i-1][j]-img1[i][j],2)+np.power(img1[i][j+1]-img1[i][j],2)+np.power(img1[i+1][j]-img1[i][j],2)
                value = abs(img1[i-1][j]-img1[i][j])**2+abs(img1[i][j+1]-img1[i][j])**2+abs(img1[i+1][j]-img1[i][j])**2
                sum ++ value
            elif i!=0 and i!=x-1 and j==y-1:  # 右中
                #value = np.power(img1[i-1][j]-img1[i][j],2)+np.power(img1[i+1][j]-img1[i][j],2)+np.power(img1[i][j-1]-img1[i][j],2)
                value = abs(img1[i-1][j]-img1[i][j])**2+abs(img1[i+1][j]-img1[i][j])**2+abs(img1[i][j-1]-img1[i][j])**2
                sum += value
            elif i==x-1 and j==0:    #左下
                #value = np.power(img1[i-1][j]-img1[i][j],2)+np.power(img1[i][j-1]-img1[i][j],2)
                value = abs(img1[i-1][j]-img1[i][j])**2+abs(img1[i][j-1]-img1[i][j])**2
                sum += value
            elif i==x-1 and j!=0 and j!=y-1:  # 下中
                #value = np.power(img1[i][j-1]-img1[i][j],2)+np.power(img1[i][j+1]-img1[i][j],2)+np.power(img1[i-1][j]-img1[i][j],2)
                value = abs(img1[i][j-1]-img1[i][j])**2+abs(img1[i][j+1]-img1[i][j])**2+abs(img1[i-1][j]-img1[i][j])**2
                sum += value
            elif i==x-1 and j==y-1:       # 右下
                #value = np.power(img1[i-1][j]-img1[i][j],2)+np.power(img1[i][j-1]-img1[i][j],2)
                value = abs(img1[i-1][j]-img1[i][j])**2+abs(img1[i][j-1]-img1[i][j])**2
                sum += value
            elif i!=x-1 and i!=0 and j!=0 and j!=y-1:                        # 中间
                #value = np.power(img1[i-1][j]-img1[i][j],2)+np.power(img1[i+1][j]-img1[i][j],2)+np.power(img1[i][j-1]-img1[i][j],2)+np.power(img1[i][j+1]-img1[i][j],2)
                value = abs(img1[i-1][j]-img1[i][j])**2+abs(img1[i+1][j]-img1[i][j])**2+abs(img1[i][j-1]-img1[i][j])**2+abs(img1[i][j+1]-img1[i][j])**2
                sum += value
    sum1 = 3*(y-2)+4+(y-2)*4*(x-2)+2*3*(x-2)
    print('loss',sum/sum1)
    print('sum',sum)

    print(x)
    print(y)
if __name__ == '__main__':
    due4(img)
