import cv2
import imutils
import numpy as np
import os
from matplotlib import pyplot as plt


def cv_show(name, img):  # 定义一个函数，显示图片
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
def findDes(images):  # 寻找模板特征值
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)  # 添加一个新的特征值
    return desList


def findSpots(images):  # 寻找模板特征点
    spotList = []
    for img in images:
        kp, des = orb.compute(img, None)
        spotList.append(kp)  # 添加一个新的特征值
    return spotList


def findID(img, desList, thres=15):  # 匹配是否存在匹配
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatchercreate(cv2.NORM_HAMMING)
    matchList = []
    finalVal = -1  # 设成-1,这样如果第一张图匹配成功就是finalVal==0,classNames[0]即可输出第一张照片名称
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []  # 从大到小的最佳匹配
            for m, n in matches:     # 对knn计算结果进行限制来排除假匹配
                if m.distance < 0.9 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    if len(matchList) != 0:
        # 匹配到的点大于阈值，判断为图片内容相同
        if max(matchList) > thres:  # 最匹配的一张至少超过thres个相似点
            finalVal = matchList.index(max(matchList))  # max输出匹配最成功的是第几个
    return finalVal


def findIDSP(img, desList):  # 匹配是否存在匹配
    kp2, des2 = orb.compute(img, None)
    bf = cv2.BFMatcher.create(cv2.NORM_HAMMING)
    matchList = []
    finalVal = -1  # 设成-1,这样如果第一张图匹配成功就是finalVal==0,classNames[0]即可输出第一张照片名称
    try:
        for des in desList:
            matches = bf.match(des, des2)
            min_distance = matches[0].distance
            max_distance = matches[0].distance
            for x in matches:
                if x.distance < min_distance:
                    min_distance = x.distance
                if x.distance > max_distance:
                    max_distance = x.distance
            good = []  # 从大到小的最佳匹配
            for x in matches:  # 对knn计算结果进行限制来排除假匹配
                if x.distance <= max(2 * min_distance, 30):
                    good.append(x)
            matchList.append(len(good))
    except:
        pass
    if len(matchList) != 0:
        finalVal = matchList.index(max(matchList))  # max输出匹配最成功的是第几个
    return finalVal


# orb准备 储存模板特征值 储存照片名
orb = cv2.ORB.create(nfeatures=5000)
path = 'Number'
images = []
classNames = []
myList = os.listdir(path)  # 返回指定路径下的文件和文件夹列表
for n in myList:
    e_img = cv2.imread(f'{path}/{n}', 0)
    images.append(e_img)
    classNames.append(os.path.splitext(n)[0])  # 把所有照片名（path路径下）放入数组，匹配成功即可输出
desList = findDes(images)
# 只好再写个找点惹呜呜呜
for n in myList:
    e_img = cv2.imread(f'{path}/{n}', 0)
    images.append(e_img)
spotList = findSpots(images)
'''

path = 'Number'
images = []
classNames = []
mylist = os.listdir(path)

for n in mylist:
    e_img = cv2.imread(f'{path}/{n}', 0)
    images.append(e_img)
    classNames.append(os.path.splitext(n)[0])

img = cv2.imread('real12.jpg')
# 滤波
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化灰度图
# apimg = cv2.adaptiveThreshold(imggray, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
retval, dst = cv2.threshold(imggray, 100, 255, cv2.THRESH_TOZERO_INV)  # 二值化
retval1, dst1 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV)  # 二值化
# cv_show(' ', dst1)
mask = cv2.erode(dst1, None, iterations=7)  # 腐蚀
mask = cv2.dilate(mask, None, iterations=2)  # 膨胀
# cv_show('mask', mask)
imgcanny = cv2.Canny(mask, 50, 100)  # canny轮廓检测
# cv_show('canny', imgcanny)

# 找出数字
_, fcon, hier = cv2.findContours(imgcanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
# draw = cv2.drawContours(img, fcon, -1, (0, 0, 255), 1)  # 绘制轮廓
# cv_show('   ', draw)
ij = 0  # angry! 每个roi都有两遍！只好判断奇偶性55
digit_out = []
locate = []
oknum = -1
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 画出所有矩形框
    if w < 50 or h < 100:  # 寻找包围数字矩形
        continue
    if w > 250 or h > 250:
        continue
    ij = ij + 1
    if ij % 2 == 1:
        # print('w,h:', w, h)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)  # 插眼
        r_img = mask[y:y + h, x:x + w]  # 截取出每个ROI区域
        # cv_show('roii', r_img)
        pro = float(150 / h)
        reimg = cv2.resize(r_img, (0, 0), fx=pro, fy=pro, interpolation=cv2.INTER_AREA)
        # cv_show('re', reimg)
        source = []
        digit = 0
        for n in images:
            res = cv2.matchTemplate(reimg, n, cv2.TM_CCOEFF_NORMED)  # 1：完美匹配 -1：不匹配
            max_val = cv2.minMaxLoc(res)[1]
            source.append(max_val)
        digit = source.index(max(source))
        name = classNames[digit]
        if max(source) > 0.5:
            locate.append(x)
            digit_out.append(name)
            for r in range(len(locate)-1):
                for i in range(r+1,len(locate)):
                    if locate[r] > locate[i]:
                        locate[r], locate[i] = locate[i], locate[r]
                        digit_out[r], digit_out[i] = digit_out[i], digit_out[r]
print(digit_out)

'''finalVal = matchList.index(max(matchList))
            loc = np.where(res >= thre)
            h1, w1 = n.shape[:2]
            for pt in zip(*loc[::-1]):
                cv2.rectangle(reimg, pt, (pt[0] + w1, pt[1] + h1), (0, 0, 150), 2)
        cv_show('?', reimg)
'''

# cv_show('out', img)  # 插眼

# 模板：带有角度 调整灰度值 对roi变换成和模板同样尺寸


'''
orb = cv2.ORB.create(nfeatures=1000)
path = 'images'
images = []
classNames = []
# 读取图片
myList = os.listdir(path)
for n in myList:
    img = cv2.imread('{path}/{n}', 0)
    images.append(img)
    classNames.append(os.path.splitext(n)[0])
desList = findDes(images)

cap = cv2.VideoCapture(1)
while True:
    _, img = cap.read()
    imgCopy = img.copy()
    # 转换成灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    id = findID(img, desList)
    if id != -1:
        cv2.putText(imgCopy, classNames[id],
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('img', imgCopy)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyWindow()


# 用面积、轮廓长度判断数字     
# dcon = cv2.drawContours(r_img, c, -1, (255, 255, 255), 1)  # 绘制轮廓
# cv_show('?', dcon)
# S = cv2.contourArea(c)  # 面积判断  7:9584/9453/9910 5:15497.5/15755 8:24058/24564 6:21932/22970
# print(S)
'''
