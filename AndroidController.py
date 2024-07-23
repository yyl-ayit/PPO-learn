import subprocess
import time

import scrcpy
from adbutils import adb

import Myclass as My
from config import appName, screenCentre, gameName


class AndroidController:
    device_id = "127.0.0.1:5555"  # 这个是蓝叠安卓模拟器特有的，如果是其他模拟器，需要调整
    app_package_name = appName

    def __init__(self):
        self.frameImg = None
        self.clientMain = None

    @staticmethod
    def getImgBinary():
        result = subprocess.run(["adb", "connect", "127.0.0.1:5555"])

        if result.returncode == 0:
            print("连接成功")
        else:
            raise ValueError('无法链接到设备， 请手动打开模拟器。')

        process = subprocess.Popen(
            ["scrcpy", "--render-expired-frames", "-s", AndroidController.device_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process.stdout.read()

    def on_client_frame(self, frame):

        if frame is not None:
            self.frameImg = frame

    def runScrcpy(self):
        if 'cannot' in str(adb.connect("127.0.0.1:5555")):
            raise ValueError('无法链接到设备，请打开模拟器')

        print("设备连接成功")

        device_id = self.device_id
        app_package_name = self.app_package_name
        subprocess.run(["adb", "-s", device_id, "shell", "am", "start", "-n", app_package_name])
        print(f"{gameName}······")
        initModels = My.InitModel()  # 提前加载模型
        time.sleep(5)
        initModels.getReward()  # 加载回报对应的操作文件

        max_width = 1080  # 设置视频流的最大宽度为1080像素。
        max_fps = 20  # 设置视频流的最大帧率（fps）为20帧每秒。
        bit_rate = 200000  # 设置视频流的比特率为2MBps（2,000,000比特每秒）。
        client = scrcpy.Client(device=adb.device_list()[0], max_width=max_width, max_fps=max_fps, bitrate=bit_rate)
        # 为客户端添加一个监听器，用于'EVENT_FRAME'事件。
        # 当接收到帧时，`on_client_frame`方法将被调用。
        # on_client_frame： 记录实时画面
        client.add_listener(scrcpy.EVENT_FRAME, self.on_client_frame)
        # 以线程化模式启动客户端，使其在后台运行。
        # 这对于在视频流活动的同时继续执行其他任务很有用。
        client.start(threaded=True)
        self.clientMain = client
        return client, initModels

    def startGame(self):
        """
        通过ocr来识别当前的页面， 并进行导航到开始游戏
        :return: None
        """
        while True:
            if self.frameImg is not None:
                imgTxt = My.My.getText(self.frameImg)
                if 'Challenge' in imgTxt or "Tap" in imgTxt:
                    self.clientMain.control.touch(screenCentre[0], screenCentre[1], scrcpy.ACTION_DOWN)
                    self.clientMain.control.touch(screenCentre[0], screenCentre[1], scrcpy.ACTION_UP)
                    print("发现了游戏主页， 游戏开始。")
                    time.sleep(0.5)
                    return True
                elif 'Log in' in imgTxt or 'your' in imgTxt:
                    self.clientMain.control.touch(screenCentre[0] + screenCentre[0] // 2,
                                                  screenCentre[1] + screenCentre[1] - screenCentre[1] // 7,
                                                  scrcpy.ACTION_DOWN)
                    self.clientMain.control.touch(screenCentre[0] + screenCentre[0] // 2,
                                                  screenCentre[1] + screenCentre[1] - screenCentre[1] // 7,
                                                  scrcpy.ACTION_UP)
                    # print("继续游戏， 点击了【play】。")
                    time.sleep(1)
                    return True
                elif not My.My.gameEnd(self.frameImg):
                    # print("不进行复活")
                    time.sleep(3.5)
                else:
                    # 为了防止破新纪录无法退出
                    time.sleep(1)
                    self.clientMain.control.touch(screenCentre[0] * 2 - 20, screenCentre[1], scrcpy.ACTION_DOWN)
                    self.clientMain.control.touch(screenCentre[0] * 2 - 20, screenCentre[1], scrcpy.ACTION_UP)
