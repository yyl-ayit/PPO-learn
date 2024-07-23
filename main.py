import os

from predict import PPOPredict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from AndroidController import AndroidController
from train import PPOTrainer
from config import is_predict

#################################################################
# 注意，运行之前需要手动打开模拟器，安装好游戏，将config.py中对应的参数配置好。
#################################################################
Android = AndroidController()  # 类实例化
if __name__ == '__main__':
    clientMain, initModels = Android.runScrcpy()  # 启动游戏

    try:
        if is_predict:  # 主要是对模型进行预测推理
            Agent = PPOPredict()
            Agent.run(Android, initModels, clientMain)
        else:
            # 当前训练次数
            Agents = PPOTrainer(number=0)  # number：第几次训练
            # 开始训练
            Agents.run(Android, initModels, clientMain)
    finally:
        clientMain.stop()
