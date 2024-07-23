import os
import subprocess
import threading
import time

from torch.utils.tensorboard import SummaryWriter

import Myclass as My
from PPO import PPO
from config import *


class PPOTrainer(object):
    def __init__(self, number):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.ppo_agent = PPO(stateDim, actionDim, lrActor, lrCritic, gamma, KEpochs, eps_clip,
                             has_continuous_action_space,
                             actionStd)
        # Build a tensorboard
        self.writer = SummaryWriter(log_dir='runs/PPO/number_{}'.format(number))

    def run(self, Android, model, clientMain):

        try:
            time_step = 0
            i_episode = 0
            print_running_reward = 0
            print_running_episodes = 0
            while time_step <= epochs:
                Android.startGame()  # 这里主要用的ocr对画面进行识别， 然后根据返回的文字信息，判断是到那一步了， 进行对应的操作。
                current_ep_reward = 0
                done = False
                startTime = time.time()  # 记录一下开始的时间
                reward = 1
                # 每局游戏开始
                nowState = 0
                while not done:
                    if Android.frameImg is None:
                        continue
                    # reward = time.time() - startTime # 这是我做过最错的决定， 想体验的话，可以试试（将注释去掉即可）。
                    # 这里为了提高代码的速度，
                    # 我采用检测画面是否有金币图标，来进行优化， 也就是说，中用到了numpy对某块图像数据的三原色，求均值
                    if not My.My.gameEnd(Android.frameImg):  # 判断游戏结束（如果换其他游戏，需要自己改）
                        done = True
                        reward = -1  # 游戏结束， 给予负回报
                        print("用时：", time.time() - startTime)  # 打印每据可以玩多长时间
                        print_running_reward += current_ep_reward
                        print_running_episodes += 1
                        i_episode += 1  # 游戏结束， 轮数加1
                    image = Android.frameImg
                    # 这里用的retnet50进行预处理。
                    tensorImage = model.dealImage.run(My.My.dealFrame(image))  # My.My.dealFrame(image) 剪切掉无影响的图片。
                    action, nowState = self.ppo_agent.select_action(tensorImage, nowState)  # 选择动作， 用的离散ppo
                    My.My.executeOperation(action, model, clientMain)  # 后来为什么不用多线程， 因为我发现，操作频率太快ia，最起码这个游戏不需要并发。
                    # # 这里启用了线程， 这样可以实时操作
                    # action_thread = threading.Thread(target=My.My.executeOperation,
                    #                                  args=(action, model, clientMain))  # 执行操作
                    #
                    # action_thread.start()  # 启动线程
                    # 如果使用过线程， 把后面 action_thread.join() 的注释也去掉。
                    current_ep_reward += reward
                    self.ppo_agent.buffer.rewards.append(reward)  # 默认每次的回报为1, 失败后回报为0
                    self.ppo_agent.buffer.is_terminals.append(done)
                    time_step += 1
                    current_ep_reward += reward
                    # 可以理解为更新大脑
                    if time_step % update_timestep == 0:
                        print("开始进行更新······")
                        self.ppo_agent.update()
                    # 如果连续动作空间;然后衰减输出动作分布的衰减动作标准
                    if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                        self.ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                    # 保存权重
                    if time_step % save_model_freq == 0:
                        self.ppo_agent.save(saveModelPath.format(epoch=i_episode))
                    # 打印平均奖励
                    if time_step % print_freq == 0:
                        if print_running_episodes != 0:  # 可能会出现除以0的情况
                            print_avg_reward = print_running_reward / print_running_episodes

                            print_avg_reward = round(print_avg_reward, 2)
                            self.writer.add_scalar('step_rewards_{}'.format(gameName), print_avg_reward,
                                                   global_step=time_step)
                            print(
                                "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                                  print_avg_reward))

                            print_running_reward = 0
                            print_running_episodes = 0

                    # action_thread.join()  # 等待并发执行完成
                    # 用来重启游戏，防止卡顿
                    if i_episode % 1000 == 0 and i_episode != 0:
                        subprocess.run(
                            ["adb", "-s", Android.device_id, "shell", "am", "force-stop",
                             Android.app_package_name.split('/')[0]])
                        time.sleep(3)
                        # 启动应用
                        subprocess.run(
                            ["adb", "-s", Android.device_id, "shell", "am", "start", "-n", Android.app_package_name])
                        time.sleep(10)
        finally:
            self.ppo_agent.save(modelPath)  # 防止发生以外， 最后保存一下模型参数
