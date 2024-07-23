import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import Myclass as My
import cv2

# # 定义预处理转换
# preprocess = transforms.Compose([
#     transforms.Resize(224),  # 缩放图像到224x224
#     transforms.ToTensor(),   # 转换图像为PyTorch张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化处理
#     transforms.Lambda(lambda x: x.unsqueeze(0)),  # 添加额外的步骤来确保张量的形状为 (batch_size, channels, height, width)
# ])
#
# # 自定义二值化函数
# def binaryzation(image):
#     # 将图像转换为灰度图像
#     gray_image = transforms.Grayscale(1)(image)
#     # 应用自定义阈值来二值化
#     binary_image = torch.where(gray_image < 0.5, torch.tensor(0).type_as(gray_image), torch.tensor(1).type_as(gray_image))
#     return binary_image
#
#
# # 使用预处理转换处理图片
# image = Image.open('testImg/main3.jpg')
# binary_image = preprocess(image)
# binary_image = binaryzation(binary_image)
# if binary_image.ndim == 3:
#     # 如果张量是三维的，可能是彩色图像，我们需要将其转换为灰度图像
#     binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
#
# # 转换二值图像张量为OpenCV格式的图像
# binary_image_cv = binary_image.type(torch.uint8).numpy()
# My.My.debugImgShow(binary_image_cv)
if __name__ == '__main__':
    image = cv2.imread("testImg/main3.jpg")

    P = My.DealImage()
    img = P.run(image)
    print(img.shape)
