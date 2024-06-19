import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time
import os

# 定数
g = 9.80  # 重力加速度 

# リンクの長さ
l1 = 1.0
l2 = 1.0

# 重心までの長さ
lg1 = 0.5
lg2 = 0.5

# 質点の質量
m1 = 10.0
m2 = 10.0

# 慣性モーメント
I1 = m1 * l1**2 / 3
I2 = m2 * l2**2 / 3

# 粘性係数
b1 = 0.5
b2 = 0.5

# 初期条件
q10 = -np.pi / 8
q20 = 0
q1_dot0 = 0.0
q2_dot0 = 0.0

dt = 0.01

# CSVファイルの保存先ディレクトリ
save_dir = r'data5'

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 運動方程式（オイラー法）
def update_world(q1, q2, q1_dot, q2_dot, action):
    # 行動に基づく外力を設定
    F = np.zeros((2, 1))
    if action == 0:
        F = np.array([[1.0], [0.0]])
    elif action == 1:
        F = np.array([[-1.0], [0.0]])
    elif action == 2:
        F = np.array([[0.0], [1.0]])
    elif action == 3:
        F = np.array([[0.0], [-1.0]])
    elif action == 4:
        F = np.array([[1.0], [1.0]])
    elif action == 5:
        F = np.array([[-1.0], [-1.0]])
    elif action == 6:
        F = np.array([[1.0], [-1.0]])
    elif action == 7:
        F = np.array([[-1.0], [1.0]])
    elif action == 8:
        F = np.array([[0.0], [0.0]])

    # 質量行列
    M_11 = m1*lg1**2 + I1 + m2*(l1**2 + lg2**2 + 2*l1*lg2*np.cos(q2)) + I2
    M_12 = m2 * (lg2**2 + l1 * lg2*np.cos(q2)) + I2
    M_21 = m2 * (lg2**2 + l1*lg2 * np.cos(q2)) + I2
    M_22 = m2 * lg2**2 + I2

    M = np.array([[M_11, M_12],
                  [M_21, M_22]])

    # コリオリ行列
    C_11 = -m2 * l1 * lg2 * np.sin(q2) * q2_dot * (2 * q1_dot + q2_dot)
    C_21 = m2 * l1 * lg2 * np.sin(q2) * q1_dot**2
    C = np.array([[C_11], [C_21]])

    # 重力ベクトル
    G_11 = m1 * g * lg1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + lg2 * np.cos(q2))
    G_21 = m2 * g * lg2 * np.cos(q2)
    G = np.array([[G_11], [G_21]])

    # 粘性
    B_11 = b1 * q1_dot
    B_21 = b2 * q2_dot
    B = np.array([[B_11], [B_21]])

    # 逆行列
    M_inv = np.linalg.inv(M)

    # 運動方程式（オイラー法）
    q_dot_dot = M_inv.dot(F - C - G - B)
    q1_dot_dot = q_dot_dot[0, 0]
    q2_dot_dot = q_dot_dot[1, 0]

    # オイラー法
    q1_new = q1 + q1_dot * dt
    q2_new = q2 + q2_dot * dt
    q1_dot_new = q1_dot + q1_dot_dot * dt
    q2_dot_new = q2_dot + q2_dot_dot * dt

    #  # 角度制約
    # q2_new = np.clip(q2_new, 0, 29 * np.pi / 36)
    return q1_new, q2_new, q1_dot_new, q2_dot_new

max_number_of_steps = 6000 # 最大ステップ数
num_episodes = 10

# Q = np.random.uniform(low = 1, high = 1, size = (30**30, 9))
# Qテーブルの初期化
num_q1_bins = 30
num_q2_bins = 30
num_q1_dot_bins = 30
num_q2_dot_bins = 30
num_actions = 9  # 行動数

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins, num_actions))

# binの設定（等間隔数列で返す）
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(q1, q2, q1_dot, q2_dot):
    digitized = [
        np.digitize(q1, bins = bins(-np.pi, np.pi, 30)) - 1,
        np.digitize(q2, bins = bins(0, 29*np.pi / 36, 30)) - 1,
        np.digitize(q1_dot, bins = bins(-30, 30, 30)) - 1,
        np.digitize(q2_dot, bins = bins(-3.0, 3.0, 30)) - 1
    ]

    # return q1_bin, q2_bin, q1_dot_bin, q2_dot_bin
    return tuple(digitized)

# ε-greedy法に基づく行動の選択
def get_action(state, episode):
    epsilon = 0.5 * (0.99 ** episode)
    if epsilon <= np.random.uniform(0, 1):
        action = np.argmax(Q[state])
    else:
        action = np.random.choice(num_actions)
    return action


def update_q_table(state, action, reward, next_state, next_action, alpha = 0.2, gamma = 0.99):
    Q[state][action] = (1 - alpha) * Q[state][action] + \
                        alpha * (reward + gamma * Q[next_state][next_action])

# リセット関数
def reset():
    q1 = -np.pi / 8  # 初期角度 q1
    q2 = 0  # 初期角度 q2
    q1_dot = 0.0  # 初期角速度 q1_dot
    q2_dot = 0.0  # 初期角速度 q2_dot
    return q1, q2, q1_dot, q2_dot

# Q学習のメイン関数
def q_learning(update_world):
    step_list = []
    for episode in range(num_episodes):
        q1, q2, q1_dot, q2_dot = reset()

        state = digitize_state(q1, q2, q1_dot, q2_dot)
        action = get_action(state, episode)

        episode_reward = 0

        csv_file_path = os.path.join(save_dir, f'data1_episode_{episode + 1}.csv.csv')
        with open(csv_file_path, 'a', newline='') as csvfile:  # 'w'モードで追記
            csv_writer = csv.writer(csvfile)
            if os.stat(csv_file_path).st_size == 0:  # ファイルが空の場合、ヘッダーを書き込む
                csv_writer.writerow(['episode', 'Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'Reward'])

            for i in range(max_number_of_steps):
                next_q1, next_q2, next_q1_dot, next_q2_dot = update_world(q1, q2, q1_dot, q2_dot, action)

                if next_q1 > q1:
                    reward = 1
                else:
                    reward = -1

                next_state = digitize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)
                next_action = get_action(next_state, episode)

                update_q_table(state, action, reward, next_state, next_action)

                state, action = next_state, next_action
                q1, q2, q1_dot, q2_dot = next_q1, next_q2, next_q1_dot, next_q2_dot
                episode_reward += reward

                csv_writer.writerow([episode + 1, i * dt, q1, q2, q1_dot, q2_dot, episode_reward])

                print(f'episode: {episode + 1}, Step: {i}, Total Reward: {episode_reward}, Q1: {q1}, Q2: {q2}, Q1_dot: {q1_dot}, Q2_dot: {q2_dot}, Action: {action}')

                time.sleep(0.01)

        print(f'Data for episode {episode + 1} has been saved to {csv_file_path}')
            

if __name__ == "__main__":
    q_learning(update_world)
