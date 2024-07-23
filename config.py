gameName = "地铁酷跑"
epochs = int(2e8)  # 训练轮数（这里和总操作次数继续来继续结束的判断， 并不是游戏的轮数）
lamda = 0.99  # 较大的λ会给优势函数更多的权重，而较小的λ则会给价值函数更多的权重

hidden_width = 64  # 决定了隐藏层中神经元的数量
print_freq = int(2e2)  # 打印的频率
actionStd = 0.1  # 随机的可能性
# o.6
save_model_freq = int(1e4)  # 保存频率
KEpochs = 80  # 更新的轮数
update_timestep = 510
eps_clip = 0.1  # 大胆性 建议[0.1, 0.3]
gamma = 0.999  # gamma 接近1意味着智能体更看重长期回报，而 gamma 接近0则意味着智能体更看重即时回报.

lrActor = 0.0003  # 演员网络学习率
lrCritic = 0.001  # 评论家网络学习率

# 必须与 ImageFeatureVectors 的值一样
# 需要理解，这里我将当前的游戏画面的特征向量作为游戏特征参数空间
stateDim = 64  # 特征参数空间
# 不建议调太大，特征参数空间越大，往往对应的情况越多，需要花费的时间和算力就越多。

# 请务必要和toOperation.json中的操作数量一样
actionDim = 5  # 操作数量
action_std_decay_freq = int(2.5e5)  # 包括下面的参数，可以看原作者注释， 因为本项目用不到， 就不做详细解释了
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
ImageFeatureVectors = 64  # 图片预处理后的图片特征向量大小 而且必须与 actionDim 的值一样, 可以看testImageDeal.py处理的结果
# 可以在模拟器中的设置-> 显示-> 分辨率中找到， 我的是(960, 540), 所以就是（270, 480）,分别代表（x, y）
screenCentre = (270, 480)  # 一定要根据自己的实际情况进行设置， 对应的上滑，下滑， 左滑， 右滑分别将对应的坐标减100， 如：上滑： 从中心移动到（260， 504）
is_predict = True  # 是否进行预测测试， 默认是读出 modelPath = "model/lastModel.pth" 地址
has_model = True  # 如果想要加训，改这个
has_continuous_action_space = False  # 是否为连续动作，本项目因为每次就做出一个动作，故选择离散的
directory = "model"  # 模型保存文件夹
modelPath = "model/lastModel.pth"  # 模型保存地址， 最后保存
saveModelPath = "model/Model{epoch}.pth"  # 模型保存地址，特定频率保存
appName = "com.kiloo.subwaysurf/com.kiloo.subwaysurf.RRAndroidPluginActivity"  # 如果不知道怎么获得：windows:adb -s 127.0.0.1:5555 shell dumpsys window | findstr mCurrentFocus
# Unix-like: adb -s 127.0.0.1:5555 shell dumpsys window | grep mCurrentFocus

# tensorboard --logdir=runs/PPO # 运行查看整体趋势
