---
layout: post
title:  "强化学习之q-learning"
date:   2025-03-28 14:30:00 +0800
tags: [linux]
pin: true
math: true
---


转载请注明出处：

<https://darkknightzh.github.io/posts/qlearning>


参考网址：

<https://www.cnblogs.com/chase-youth/p/18110374>

<https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial>


<https://zhuanlan.zhihu.com/p/74065749>

<br>

## P1. 简介


Q-learning是一种无模型、基于值的非策略算法，其根据agent的当前状态找到最佳行为。“Q”意为quality，代表了action在最大化未来回报时的价值。

- 基于模型的算法使用转换和奖励函数来估计最优策略并创建模型。
- 无模型的算法通过（没有转换和奖励函数的）经验来学习对应行为的后果。
- 基于值的方法训练价值函数，来学习哪种状态更有价值并采取行为。
- 基于策略的方法直接训练策略，以学习在给定状态下应采取的行为。


## P2. 基本概念

- Agent：在环境中执行动作的实体。
- Environment（环境）：agent所处并与之交互的外部世界。
- State（状态）：agent在环境中的当前位置（或状态），表示为 S。
- Action（行为）：agent在特定状态下采取的行为，表示为 A。
- Reward（奖励）：执行一个行为后，环境反馈给agent的奖励或惩罚，表示为 R。
- Policy（策略）：从状态到行为的映射，即智能体在某状态下应采取的行为。
- Q-value（Q值）：在某状态下采取行为期望得到的回报。
- $Q({S}_{t+1}, a)$：在特定状态下采取行为期望得到的最优回报。
- Q-Table：存储状态和行为对应回报的二维表。每行代表每个状态对应的所有行为（上、下、左、右移动）；每列代表每个行为对应的所有状态（开始/Start、空闲/Idle、掉入洞中/Hole、结束/End），如图1所示。图片来自<https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial>。
- Temporal Differences(TD)：用于通过当前状态和行为以及先前状态和行为来估计$Q({S}_{t+1}, a)$的期望值。

![1](/assets/post/2025-03-28-qlearning/qtable.png)
_图1-q-table_


### P2.1 $\epsilon$ 贪婪策略

$\epsilon$ 贪婪策略是一个用于平衡探索模式（exploration）和开发模式（exploitation）的方法。其中 $\epsilon$ 代表阈值。开始时 $\epsilon$ 概率较高，代表agent处于探索模式。随着迭代的进行，epsilon逐渐降低，agent开始开发环境，并且对估计Q值更加自信。
探索模式是指选择之前没执行过的行为，从而探索更多的可能性，并根据这些随机行为来得到对回报的影响；开发模式是指选择选择已经执行过的行为，从而对已知行为的模型进行完善，这些执行过的行为会使回报最大化。

对应代码：
```python
def choose_action(env, qtable, epsilon, state):
    if random.uniform(0, 1) < epsilon:
        # 探索模式，随机选择行为
        return env.action_space.sample()
    else:
        # 开发模式，选择状态state中的最大回报对应的行为
        return np.argmax(qtable[state, :])
```

### P221 更新公式

q table更新公式如下：

$$Q({S}_{t}, {A}_{t}) = Q({S}_{t}, {A}_{t}) + \alpha [{R}_{t+1} + \gamma max_{a}Q(S_{t+1}, a) - Q(S_t, A_t)]$$

其中 $\alpha$为学习率，决定新信息覆盖旧信息的速度。$ \gamma $为折扣因子，决定未来奖励的重要性。如图2所示。图片来自<https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial>。

![1](/assets/post/2025-03-28-qlearning/qlearningequation.png)
_图2-q-table更新公式_

对应代码：

```python
def update_q_table(qtable, alpha, gamma, state, action, reward, next_state):
    best_next_action = np.argmax(qtable[next_state, :])  # 下一个状态下的每个行为对应的回报的最大值对应的行为，即下一个状态的最优行为
    td_target = reward + gamma * qtable[next_state, best_next_action]  # 目标回报 = 当前行为的回报 + 折扣因子 * 下一个状态的最优回报
    td_error = td_target - qtable[state, action]  # 目标回报和当前回报的差值
    qtable[state, action] += alpha * td_error  # 更新回报公式。
    return qtable
```

## P3. 代码

```python

import numpy as np
import random
import argparse
import gym
import imageio


# q table是一个二维矩阵，行代表所有状态，列代表所有行为
def initialize_q_table(state_space, action_space):
    qtable = np.zeros((state_space, action_space))    # 行代表状态，列代表行为

    print("There are %d possible states".format(state_space))
    print("There are %d possible actions".format(action_space))
    return qtable


# 使用epsilon贪婪策略来平衡探索和开发模式
def choose_action(env, qtable, epsilon, state):
    if random.uniform(0, 1) < epsilon:
        # 探索模式，随机选择行为
        return env.action_space.sample()
    else:
        # 开发模式，选择状态state中的最大回报对应的行为
        return np.argmax(qtable[state, :])


# 通过对应公式，更新q table
def update_q_table(qtable, alpha, gamma, state, action, reward, next_state):
    best_next_action = np.argmax(qtable[next_state, :])  # 下一个状态下的每个行为对应的回报的最大值对应的行为，即下一个状态的最优行为
    td_target = reward + gamma * qtable[next_state, best_next_action]  # 目标回报 = 当前行为的回报 + 折扣因子 * 下一个状态的最优回报
    td_error = td_target - qtable[state, action]  # 目标回报和当前回报的差值
    qtable[state, action] += alpha * td_error  # 更新回报公式。
    return qtable


def train(args, env, qtable):
    epsilon = args.epsilon

    # Training
    for episode in range(args.training_episodes): # We run multiple episodes to allow the agent to learn from different starting positions.
        # state, info = env.reset()   # 新版本gym用这个代码，如1.0
        state = env.reset()  # 旧版本gym用这个代码，如0.24

        done = False
        
        for _ in range(args.max_steps):  # For each step in an episode, the agent chooses an action, observes the outcome, and updates the Q-table.
            action = choose_action(env, qtable, epsilon, state)

            # next_state, reward, terminated, truncated, _ = env.step(action)   # 新版本gym用这个代码，如1.0
            # done = terminated or truncated

            next_state, reward, done, info = env.step(action)   # 旧版本gym用这个代码，如0.24
            qtable = update_q_table(qtable, args.alpha, args.gamma, state, action, reward, next_state)
            state = next_state
            
            if done:
                break
        
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)   # Decay epsilon
    
    return qtable


def test(args, env, qtable):

    for episode in range(args.eval_episodes):
        # state, info = env.reset()   # 新版本gym用这个代码，如1.0
        state = env.reset()  # 旧版本gym用这个代码，如0.24

        done = False
        total_rewards = 0
        
        for _ in range(args.max_steps):
            action = np.argmax(qtable[state, :])

            # next_state, reward, terminated, truncated, _ = env.step(action)  # 新版本gym用这个代码，如1.0
            # done = terminated or truncated

            next_state, reward, done, info = env.step(action)   # 旧版本gym用这个代码，如0.24
            
            total_rewards += reward
            state = next_state
            
            if done:
                print(f"Episode: {episode + 1}, Total Reward: {total_rewards}")
                break
    

def record_video(env, qtable, out_directory, fps=1):
    images = []
    done = False
    img = env.render(mode='rgb_array')
    images.append(img)

    # state, info = env.reset()   # 新版本gym用这个代码，如1.0
    state = env.reset(seed=random.randint(0,500))  # 旧版本gym用这个代码，如0.24

    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state][:])

        # state, reward, terminated, truncated, _ = env.step(action)  # 新程序用这个代码，如1.0
        # done = terminated or truncated

        state, reward, done, info = env.step(action)   # 旧版本gym用这个代码，如0.24   We directly put next_state = state for recording logic

        img = env.render(mode='rgb_array')
        images.append(img)
    
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def main(args):

    if args.env_id == 'FrozenLake-v1':
        env = gym.make(args.env_id, map_name="4x4",is_slippery=False)   # FrozenLake-v1需要设置is_slippery=False，否则效果会差很多
    else:
        env = gym.make(args.env_id)
    
    q_table = initialize_q_table(env.observation_space.n, env.action_space.n)

    q_table = train(args, env, q_table)
    test(args, env, q_table)
    
    video_path="replay.gif"
    video_fps=5
    record_video(env, q_table, video_path, video_fps)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'q learning example')

    parser.add_argument('--env_id', default='Taxi-v3', type=str, required=False, help='env_id')  # 感觉最后生成的gif有点怪怪的
    # parser.add_argument('--env_id', default='FrozenLake-v1', type=str, required=False, help='env_id')

    parser.add_argument('--alpha', default=0.7, type=float, required=False, help='Learning rate, 学习率 \
                        Determines to what extent newly acquired information overrides old information.')
    parser.add_argument('--gamma', default=0.95, type=float, required=False, help='Discount factor, 折扣因子 \
                        Measures the importance of future rewards')
    
    parser.add_argument('--epsilon', default=1.0, type=float, required=False, help='Exploration rate (epsilon-greedy),\
                        Controls the trade-off between exploration (trying new actions) and exploitation (using known actions)')
    parser.add_argument('--epsilon_decay', default=0.9995, type=float, required=False, help='epsilon_decay')
    parser.add_argument('--epsilon_min', default=0.01, type=float, required=False, help='epsilon_min')

    parser.add_argument('--training_episodes', default=10000, type=int, required=False, help='Total training episodes')
    parser.add_argument('--eval_episodes', default=50, type=int, required=False, help='Total training episodes')
    parser.add_argument('--max_steps', default=100, type=float, required=False, help='Max steps per episode')

    args = parser.parse_args()

    main(args)
```


