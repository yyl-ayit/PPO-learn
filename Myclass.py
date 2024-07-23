import json
import time
from io import BytesIO

import cv2
import numpy as np
import pytesseract
import scrcpy
import torch
import torch.jit
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from config import ImageFeatureVectors, screenCentre, stateDim


class CustomModel(nn.Module):  # 可以自主定义生成的特征向量大小
    """
    为了自主定义生成的特征向量大小
    """

    def __init__(self, num_features=ImageFeatureVectors):
        if ImageFeatureVectors != stateDim:
            raise ValueError('ImageFeatureVectors 和 stateDim 的值必须相等！')
        super(CustomModel, self).__init__()
        # 加载预训练的ResNet-152模型，但不包括最后的全连接层
        self.base_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Identity()  # 移除原始的全连接层

        # 假设我们想要增加最后一个卷积层的通道数
        self.modified_layer4 = self.modify_last_layer(self.base_model.layer4, num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def modify_last_layer(layer, num_features):
        # 复制最后一层并修改输出通道数
        modified_layer = nn.Sequential(
            *(list(layer.children())[:-1]),  # 获取除了最后一个块以外的所有块
            nn.Conv2d(in_channels=layer[-1].conv3.out_channels,
                      out_channels=num_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        )
        return modified_layer

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.modified_layer4(x)  # 使用修改后的层4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class DealImage(object):
    """
    图片预处理，将图片转换成对应的特征参数空间
    """

    def __init__(self):
        # 定义预处理步骤
        self.preprocess = transforms.Compose([
            transforms.Resize(224),  # 缩放图像到224x224
            transforms.ToTensor(),  # 转换图像为PyTorch张量，形状为 (channels, height, width)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化处理
            transforms.Lambda(lambda x: x.unsqueeze(0)),  # 添加额外的步骤来确保张量的形状为 (batch_size, channels, height, width)
        ])
        # self.model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        self.model = CustomModel()
        # self.model.eval()

    # 定义一个转换，将图片二值化
    def run(self, image):
        """

        :param image: numpy_array, 图片数据
        :return: ResNet101_Weights处理后的特征向量
        """
        # 加载图像并预处理
        # 将灰度图像转换为三通道RGB图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        tensor_img = self.preprocess(Image.fromarray(image_rgb))  # 将图像转换为PIL图像，然后进行预处理

        # 使用模型提取特征
        with torch.no_grad():
            features = self.model(tensor_img)

        # 使用全局平均池化
        return torch.flatten(features, 1)


class Model(torch.jit.ScriptModule):
    """
    :return 自定义Model
    """

    def __init__(self):
        super(Model, self).__init__()

        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        self._hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

    @torch.jit.script_method
    def forward(self, x):
        x = self._hidden1(x)
        x = self._hidden2(x)
        x = self._hidden3(x)
        x = self._hidden4(x)
        x = self._hidden5(x)
        x = self._hidden6(x)
        x = self._hidden7(x)
        x = self._hidden8(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._hidden9(x)
        x = self._hidden10(x)

        length_logits = self._digit_length(x)
        digit1_logits = self._digit1(x)
        digit2_logits = self._digit2(x)
        digit3_logits = self._digit3(x)
        digit4_logits = self._digit4(x)
        digit5_logits = self._digit5(x)

        return length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits

    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = 95000
        return step


class My:
    @staticmethod
    def dealFrame(frame):
        """
        :param frame:  图片数据
        :type frame: <class 'numpy.ndarray'>
        :return: 返回处理后的图片数据
        """
        # 获取图片的高度和宽度
        height, width = frame.shape[:2]

        crop_height = int(height / 3)

        # 计算裁剪的起始和结束坐标（只对上下进行处理）
        start_y = crop_height // 7 * 6
        end_y = height - crop_height // 6 * 7

        # 截取图片中间三分之一的部分（保持左右边界不变）
        frame = frame[start_y:end_y, :]
        # 将图像转换为灰度图像
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 设置阈值和最大值
        threshold_value = 127  # 介于0到255之间，可以根据需要进行调整
        max_value = 255

        # 应用二值化
        _, binary_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
        return binary_image

    @staticmethod
    def getText(image):
        """
        :param image:  图片数据
        :type image: <class 'numpy.ndarray'>
        :return: 从图片中识别到的文字结果， 因为速度太慢，不建议应用在实时任务中
        """
        return pytesseract.image_to_string(image, config='--psm 11')

    @staticmethod
    def getNum(image):
        """
       :param image:  图片数据
       :type image: <class 'numpy.ndarray'>
       :return: 从图片中识别到的数字结果， 因为速度太慢，不建议应用在实时任务中
       """
        num = "0"
        for i in pytesseract.image_to_string(image, config='--psm 6'):
            if i.isdigit():
                num += i
        return int(num)

    @staticmethod
    def executeOperation(action, initModel, clientMain):
        """

        :param action: 对应toOperation.json中的操作
        :param initModel: 这里存放了对应的操作字典
        :param clientMain: 主客户端。
        :return: 返回None, 只执行对应的操作
        """
        if action is None:
            raise ValueError("请检查action的类型， 必须是整数!")
        if action != 0:
            operation = eval(initModel.operation.get(initModel.toOperation.get(str(action))))  # 得到对应的操作

            assert type(operation) is type((0, 0))
            endX = screenCentre[0] + operation[0]  # 通过加上对应值，来得到目的坐标。 如果学过深搜， 可能一下就懂了。
            endY = screenCentre[1] + operation[1]
            # 模拟 滑动
            clientMain.control.touch(screenCentre[0], screenCentre[0], scrcpy.ACTION_DOWN)
            time.sleep(0.2)
            clientMain.control.touch(endX, endY, scrcpy.ACTION_MOVE)
            time.sleep(0.2)  # 等待一段时间以模拟移动动作
            clientMain.control.touch(endX, endY, scrcpy.ACTION_UP)
        else:
            # 对应的无操作，不需要操作，但得和其他操作一样， 花费时间
            time.sleep(0.4)

    @staticmethod
    def debugImgShow(image):
        """
        :param image:图片数据
        :return: 将图片进行展示
        """
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def getReward(imageNumpy, rewardOrg):
        """

        :param imageNumpy:图片数据
        :param rewardOrg: 上一次得回报值
        :return: 用于计算每次图片中得分数， 并用误差进行纠错
        """
        reward = 0  # 当前的回馈值
        # 将分数掘图， 增加准确性
        imageScore = My.cutImage(imageNumpy, screenCentre[0] * 2 - screenCentre[0] // 2, screenCentre[1] // 25,
                                 screenCentre[0] * 2 - 10, screenCentre[1] // 9)
        # 将得到的金币截图
        imageGold = My.cutImage(imageNumpy, screenCentre[0] * 2 - screenCentre[0] // 2, screenCentre[1] // 6,
                                screenCentre[0] * 2 - int(screenCentre[0] / 6.8), int(screenCentre[1] / 4.4))

        reward += My.getNum(imageScore) + int(My.getNum(imageGold) * 1.5)
        # 进行对应得纠错
        if reward - rewardOrg < 0:
            reward = rewardOrg

        elif reward - rewardOrg >= 50:
            reward = rewardOrg
        return reward

    @staticmethod
    def gameEnd(imageNumpy):
        """

        :param imageNumpy:
        :return: 是否结束游戏 bool值
        """
        imageSave = My.cutImage(imageNumpy, screenCentre[0] - screenCentre[0] // 3,
                                screenCentre[1] - screenCentre[1] // 37 - screenCentre[1] // 9,
                                screenCentre[0] + screenCentre[0] // 3,
                                screenCentre[1] - screenCentre[1] // 20 + screenCentre[1] // 9)
        roi = imageSave
        R_mean = np.mean(roi[:, :, 0])
        G_mean = np.mean(roi[:, :, 1])
        B_mean = np.mean(roi[:, :, 2])
        lower_bound = np.array([62, 160, 85])
        upper_bound = np.array([70, 165, 90])
        # [ 65 161  89]
        pixel = np.array([int(R_mean), int(G_mean), int(B_mean)])
        # print(pixel)
        flag = (np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound))
        if flag:
            return False

        imageScore = My.cutImage(imageNumpy, screenCentre[0] * 2 - screenCentre[0] // 9, int(screenCentre[1] / 5.3),
                                 screenCentre[0] * 2 - int(screenCentre[0] / 15), int(screenCentre[1] / 4.8))

        roi = imageScore
        R_mean = np.mean(roi[:, :, 0])
        G_mean = np.mean(roi[:, :, 1])
        B_mean = np.mean(roi[:, :, 2])
        lower_bound = np.array([5, 165, 220])
        upper_bound = np.array([25, 195, 255])
        pixel = np.array([int(R_mean), int(G_mean), int(B_mean)])
        # 判断像素点是否在阈值范围内
        return np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound)

    @staticmethod
    def gameStart(imageNumpy, model):
        """

        :param imageNumpy:
        :param model: yolo模型
        :return: 是否开始游戏
        """
        if imageNumpy is None:
            return False
        result = model.run(imageNumpy).get('result')

        if len(result) >= 10:
            dic = {}
            for i in result:
                if i.get('type') in dic:
                    if i.get('type') == 'skill':
                        dic[i.get('type')].append((i.get('x'), i.get('y'), i.get('X'), i.get('Y')))
                    else:
                        dic[i.get('type')] = (i.get('x'), i.get('y'), i.get('X'), i.get('Y'))
                else:
                    if i.get('type') == 'skill':
                        dic[i.get('type')] = [(i.get('x'), i.get('y'), i.get('X'), i.get('Y'))]
                    else:
                        dic[i.get('type')] = (i.get('x'), i.get('y'), i.get('X'), i.get('Y'))
            if len(dic) < 5:
                return False
            ls = dic.get('skill')
            ls.sort(key=lambda x: x[0])  # 排序是为了确定每个技能都是什么
            dic['skill'] = ls
            model.location = dic
            return True
        return False

    @staticmethod
    def showImgBinary(imgBinary):
        """
        :param imgBinary: 二进制图像数据
        :return: 将图片数据进行展示
        """
        temp_file = BytesIO(imgBinary)

        # 使用PIL打开内存文件并显示图片
        image = Image.open(temp_file)
        image.show()

    @staticmethod
    def cutImage(img, x, y, X, Y):
        """
        :param img: 图片数据
        :param x: 矩形左上角x
        :param y: 矩形左上角y
        :param X: 矩形右下角X
        :param Y: 矩形右下角Y
        :type: img: <class 'numpy.ndarray'>, h: int, w: int , X: int , Y: int
        :return: 返回剪切后的矩形图像数据
        """

        cropped_image = img[y:Y, x:X]

        return cropped_image

    @staticmethod
    def getGetDealHelp(img, h, w, x, y):
        """
        :param img: 图片信息
        :param h: 对应图片得高
        :param w: 对应图片得宽
        :param x: 起始得x坐标
        :param y: 起始得Y坐标
        :type: img: <class 'numpy.ndarray'>, h: int, w: int , X: int , Y: int
        :return: 返回击杀， 死亡， 辅助得数量
        """

        data = []

        cutLen = w // 3
        for i in range(3):
            image = My.cutImage(img, x + i * cutLen + cutLen // 3, y, x + (i + 1) * cutLen, y + h)
            data.append(image)
        return data


class InitModel:
    def __init__(self):
        self.getNumModel = My()  # 我的类
        self.dealImage = DealImage()  # 图像预处理
        print('图片预处理模型加载完成')
        self.operation = None  # 操作对应的具体按钮
        self.toOperation = None  # 序号对应的操作

    def getReward(self):
        operationPath = 'jsonFile/Operation.json'
        toOperationPath = 'jsonFile/toOperation.json'

        with open(operationPath, 'r', encoding='utf-8-sig') as f:
            self.operation = json.load(f)
        with open(toOperationPath, 'r', encoding='utf-8-sig') as f:
            self.toOperation = json.load(f)
