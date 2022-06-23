import cv2
import os

path = 'Number'
images = []
classNames = []
mylist = os.listdir(path)

for n in mylist:
    e_img = cv2.imread(f'{path}/{n}', 0)
    images.append(e_img)
    classNames.append(os.path.splitext(n)[0])

img = cv2.imread('real11.jpg')  # 请改为捕捉到的图片
# 滤波
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化灰度图
retval, dst = cv2.threshold(imggray, 100, 255, cv2.THRESH_TOZERO_INV)  # 二值化
retval1, dst1 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV)  # 二值化
mask = cv2.erode(dst1, None, iterations=7)  # 腐蚀
mask = cv2.dilate(mask, None, iterations=2)  # 膨胀
imgcanny = cv2.Canny(mask, 50, 100)  # canny轮廓检测


# 找出数字
_, fcon, hier = cv2.findContours(imgcanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
ij = 0  # angry! 每个roi都有两遍！只好判断奇偶性55
digit_out = []
locate = []
oknum = -1
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    if w < 50 or h < 100:  # 寻找包围数字矩形
        continue
    if w > 250 or h > 250:
        continue
    ij = ij + 1
    if ij % 2 == 1:
        r_img = mask[y:y + h, x:x + w]  # 截取出每个ROI区域
        pro = float(150 / h)
        reimg = cv2.resize(r_img, (0, 0), fx=pro, fy=pro, interpolation=cv2.INTER_AREA)
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
print(''.join(digit_out))
print("\r")
