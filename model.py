import cv2
import imageio
import imutils
import numpy as np
import os


def cv_show(name, img):  # 定义一个函数，显示图片
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('numbers/8.jpg')
# h, w = img.shape[0:2]
# x = float(200/h)
# reimg = cv2.resize(img, (0, 0), fx=x, fy=x, interpolation=cv2.INTER_AREA)
# reimg = cv2.resize(img, (0, 0), fx=x, fy=x, interpolation=cv2.INTER_NEAREST)
# cv_show(' ', reimg)
# a, b = reimg.shape[0:2]
# print(a,b)

# 滤波
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化灰度图
# apimg = cv2.adaptiveThreshold(imggray, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
retval, dst = cv2.threshold(imggray, 70, 255, cv2.THRESH_TOZERO_INV)  # 二值化
retval1, dst1 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV)  # 二值化
# cv_show(' ', dst1)
mask = cv2.erode(dst1, None, iterations=7)  # 腐蚀
mask = cv2.dilate(mask, None, iterations=2)  # 膨胀
# cv_show('mask', mask)
imgcanny = cv2.Canny(mask, 50, 100)  # canny轮廓检测

_, fcon, hier = cv2.findContours(imgcanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
ij = 0  # angry! 每个roi都有两遍！只好判断奇偶性55
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 画出所有矩形框
    print(w, h)


    if w < 130 or h < 200:  # 寻找包围数字矩形 w:130-150
        continue
    if w > 170 or h > 300:
        continue
    ij = ij + 1
    # print(w, h)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)  # 插眼
    if ij % 2 == 1:
        r_img = mask[y:y + h, x:x + w]  # 截取出每个ROI区域
        # cv_show('roii', r_img)
        x = float(150 / h)
        reimg = cv2.resize(r_img, (0, 0), fx=x, fy=x, interpolation=cv2.INTER_AREA)
        cv_show('re', reimg)
        imageio.imsave('Number/8.jpg', reimg)

# cv_show(' ',img)

# imageio.imsave('imagess/.jpg', mask)'''