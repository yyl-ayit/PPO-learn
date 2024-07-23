import numpy as np

from config import screenCentre
import Myclass as My
import cv2

if __name__ == '__main__':
    # 是否复活
    img = cv2.imread('testImg/main4.jpg')
    image = My.My.cutImage(img, screenCentre[0] - screenCentre[0] // 3,
                           screenCentre[1] - screenCentre[1] // 37 - screenCentre[1] // 9,
                           screenCentre[0] + screenCentre[0] // 3,
                           screenCentre[1] - screenCentre[1] // 20 + screenCentre[1] // 9)
    roi = image
    R_mean = np.mean(roi[:, :, 0])
    G_mean = np.mean(roi[:, :, 1])
    B_mean = np.mean(roi[:, :, 2])
    lower_bound = np.array([5, 165, 220])
    upper_bound = np.array([25, 195, 255])
    pixel = np.array([int(R_mean), int(G_mean), int(B_mean)])
    print(pixel)
    # My.My.debugImgShow(image)
    # 金币
    # img = cv2.imread('testImg/play2.jpg')
    # image = My.My.cutImage(img, screenCentre[0] * 2 - screenCentre[0] // 9, int(screenCentre[1] / 5.3),
    #                        screenCentre[0] * 2 - int(screenCentre[0] / 15), int(screenCentre[1] / 4.8))
    # My.My.debugImgShow(image)
    # 分数
    # img = cv2.imread('testImg/play2.png')
    # image = My.My.cutImage(img, screenCentre[0] * 2 - screenCentre[0] // 2, screenCentre[1] // 25,
    #                        screenCentre[0] * 2 - 10, screenCentre[1] // 9)
    # My.My.debugImgShow(image)
    # img = cv2.imread('testImg/getDealHelp.jpg')
    # h, w = img.shape[:2]
    # for i in My.My.getGetDealHelp(img, h, w, 0, 0):
    #     cv2.imshow('img', i)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = cv2.imread('1.jpg')
    # print(My.HandImg().main(img))
    # img = cv2.imread('1.jpg')
    # print(My.DealImage().run(img).shape)
    # exit()
    # imageLs = My.My.getGetDealHelp(img, 19, 99, 711, 20)
    # for i in imageLs:
    #     cv2.imshow('q', i)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # dic = eval(
    #     "{'skill': [(470, 464, 529, 515), (538, 457, 595, 514), (606, 454, 685, 521), (691, 447, 763, 516), (747, 348, 821, 418), (841, 289, 926, 356)], 'vs': (625, 15, 690, 42), 'gold': (883, 25, 958, 48), 'getDealHelp': (711, 20, 812, 39), 'blood': (439, 193, 534, 206)}")
    #
    # My.My.debugImgShow('1.jpg', dic)
#
# # 假设right是一个NumPy数组
# right = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
# print(right + 2)
# # 使用argsort()获取索引，然后使用[::-1]进行降序排序
# sorted_indices = right.argsort()
# print(sorted_indices)
# # 使用排序后的索引来获取排序后的数组
# sorted_right = right[sorted_indices]
#
# print(sorted_right)  # 输出: [9 6 5 5 5 4 3 3 2 1 1]

# print(cv2.mean(None))
# P = My.RewardPredict()
# print(P.run(cv2.imread("image/img500.jpg")))
# img = cv2.imread('image/img500.jpg')
# mean_value = sum(cv2.mean(img))//3
# print(mean_value)

# filename = 'jsonFile/reward.json'
#
# with open(filename, 'r', encoding='utf-8-sig') as f:
#     operation = json.load(f)
# print(operation)
# P = My.NumPredict()
# result = P.run(img)
# print(result)
# print(type(result))
# assert type(P.run(img)) == type("str")
