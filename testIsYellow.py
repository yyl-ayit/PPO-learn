import cv2
import numpy as np
import Myclass as My
from config import screenCentre

# 读取图像
img = cv2.imread('testImg/main5.jpg')
image = My.My.cutImage(img, screenCentre[0] * 2 - screenCentre[0] // 9, int(screenCentre[1] / 5.3),
                       screenCentre[0] * 2 - int(screenCentre[0] / 15), int(screenCentre[1] / 4.8))

roi = image
# 计算RGB均值
R_mean = np.mean(roi[:, :, 0])
G_mean = np.mean(roi[:, :, 1])
B_mean = np.mean(roi[:, :, 2])

My.My.debugImgShow(image)


# 获取像素点的颜色值
def is_yellow(pixel):
    # 设定黄色的大致阈值范围，这些值可以根据实际图片进行调整
    # R: 200 - 255, G: 200 - 255, B: 50 - 100
    # [ 11 198 248]
    # [ 11 198 248]
    # [ 11 198 248]
    lower_bound = np.array([6, 190, 240])
    upper_bound = np.array([16, 210, 255])

    # 判断像素点是否在阈值范围内
    return np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound)


pixel = np.array([int(R_mean), int(G_mean), int(B_mean)])
print(pixel)
print(is_yellow(pixel))
