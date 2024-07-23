import cv2

from Myclass import DealImage

if __name__ == '__main__':
    video_capture = cv2.VideoCapture("testImg/play1.mp4")
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        exit()
    # 创建一个可以自由调整大小的窗口
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    P = DealImage()
    # 循环读取每一帧
    while True:
        # 读取一帧
        ret, frame = video_capture.read()

        # 如果读取成功，显示帧
        if ret:
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

            print(P.run(binary_image))

            cv2.imshow('Video', binary_image)

            # 按 'q' 键退出循环
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            # 读取失败，视频播放完毕
            print("Video ended or error.")
            break

    # 释放视频捕获对象并关闭所有窗口
    video_capture.release()
    cv2.destroyAllWindows()
    # 将BGR图像转换为HSV格式
