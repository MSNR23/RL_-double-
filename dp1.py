import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time

# 定数
g = 9.80  # 重力加速度 

# リンクの長さ
l1 = 1.0
l2 = 1.0

# 重心までの長さ
lg1 = 0.5
lg2 = 0.5

# 質点の質量
m1 = 1.0
m2 = 1.0

# 慣性モーメント
I1 = m1 * l1**2 / 3
I2 = m2 * l2**2 / 3

# 粘性係数
b1 = 0.1
b2 = 0.1

# 初期条件
q10 = -2 * np.pi / 3
q20 = -2  *np.pi / 3 
q1_dot0 = 0.0
q2_dot0 = 0.0

dt = 0.01

# 運動方程式（独立関数）
def update_world(t, s, action):
    q1, q2, q1_dot, q2_dot = s

    # 行動に基づく外力を設定
    F = np.zeros(2)
    if action == 0:
        F = np.array([1.0, 0.0])
    elif action == 1:
        F = np.array([-1.0, 0.0])
    elif action == 2:
        F = np.array([0.0, 1.0])
    elif action == 3:
        F = np.array([0.0, -1.0])
    elif action == 4:
        F = np.array([1.0, 1.0])
    elif action == 5:
        F = np.array([-1.0, -1.0])
    elif action == 6:
        F = np.array([1.0, -1.0])
    elif action == 7:
        F = np.array([-1.0, 1.0])
    elif action == 8:
        F = np.array([0.0, 0.0])
    
    # 質量行列
    M = np.array([[m1*lg1**2 + I1 + m2*(l1**2 + lg2**2 +2*l1*lg2*np.cos(q2))+I2, m2 * (lg2**2 +l1 * lg2*np.cos(q2))+I2],
                  [m2 * (lg2**2 + l1*lg2 * np.cos(q2)) + I2, m2 * lg2**2 + I2]])

    # コリオリ行列
    C = np.array([[-m2 * l1 * lg2 * np.sin(q2) * q2_dot * (2 * q1_dot + q2_dot)],
                  [m2 * l1 * lg2 * np.sin(q2) * q1_dot**2]])

    # 重力ベクトル
    G = np.array([[m1 * g * lg1 * np.cos(q1) + m2 * g *(l1 * np.cos(q1) + lg2 * np.cos(q1 + q2))],
               [m2 * g * lg2 * np.cos(q1 + q2)]])

    # 粘性
    B = np.array([[b1 * q1_dot],
                  [b2 * q2_dot]])

    # 逆行列
    M_inv = np.linalg.inv(M)

    # 運動方程式
    q_ddot = M_inv.dot(-C - G + B + F)

    # 角度と角速度の更新
    q1_dot_new = q1_dot + q_ddot[0, 0] * dt
    q2_dot_new = q2_dot + q_ddot[1, 0] * dt
    q1_new = q1 + q1_dot_new * dt
    q2_new = q2 + q2_dot_new * dt

    # 更新した値を返す
    return np.array([q1_new, q2_new, q1_dot_new, q2_dot_new])

    # return np.array([q1_dot, q2_dot, q_ddot[0, 0], q_ddot[1, 0]])

# Runge-Kutta法
def runge_kutta(t, s, F, dt):
    k1 = dt * update_world(t, s, F)
    k2 = dt * update_world(t + 0.5 * dt, s + 0.5 * k1, F)
    k3 = dt * update_world(t + 0.5 * dt, s + 0.5 * k2, F)
    k4 = dt * update_world(t + dt, s + k3, F)

    s_new = s + (k1 + 2*k2 + 2*k3 + k4) / 6

    return s_new

# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.5  # ε-greedy法のε

# Qテーブルの初期化
num_q1_bins = 10
num_q2_bins = 10
num_q1_dot_bins = 10
num_q2_dot_bins = 10
num_actions = 9

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins,num_actions))

# 状態の離散化関数
def discretize_state(s):
    q1_bin = np.linspace(-np.pi, np.pi, 11)  # q1 のビン境界
    q2_bin = np.linspace(0, 29 * np.pi / 36, 10)  # q2 のビン境界
    q1_dot_bin = np.linspace(-6.0, 6.0, 12)  # q1_dot のビン境界
    q2_dot_bin = np.linspace(-6.0, 6.0, 12)  # q2_dot のビン境界

    #  # binが範囲内に収まるように調整
    # q1_bin = max(0, min(num_q1_bins - 1, q1_bin))
    # q2_bin = max(0, min(num_q2_bins - 1, q2_bin))
    # q1_dot_bin = max(0, min(num_q1_dot_bins - 1, q1_dot_bin))
    # q2_dot_bin = max(0, min(num_q2_dot_bins - 1, q2_dot_bin))

    # return q1_bin, q2_bin, q1_dot_bin, q2_dot_bin
    state = [
        np.digitize(s[0], q1_bin) - 1,  # np.digitize は1から始まるので調整
        np.digitize(s[1], q2_bin) - 1,
        np.digitize(s[2], q1_dot_bin) - 1,
        np.digitize(s[3], q2_dot_bin) - 1
    ]

    return tuple(state)

# リセット関数
def reset():
    q1 = q10
    q2 = q20
    q1_dot = q1_dot0
    q2_dot = q2_dot0

    return q1, q2, q1_dot, q2_dot

# ε-greedy法に基づく行動の選択
def get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, :])

# # ε-greedy法に基づく行動の選択
# def get_action(theta_bin, omega_bin):
#     if np.random.rand() < epsilon:
#         return np.random.choice(num_actions + 1)
#     else:
#         return np.argmax(Q[theta_bin, omega_bin, :])

# 前回の報酬を格納する変数
# prev_reward = 0

# 報酬を計算する関数
def calculate_reward(s, prev_s):
    # global prev_reward  # グローバル変数として前回の報酬を使用することを明示

    # 前の状態と比較してq1_dotの増加量が正の場合に報酬を与える
    delta_q1_dot = s[2] - prev_s[2]
    if delta_q1_dot > 0:
        reward = 1  # 増加した場合の報酬
    else:
        reward = -100  # それ以外の場合は報酬を与えない or ペナルティを与える
    
    #     # 前回の報酬と今回の報酬を累積して加算
    # total_reward = prev_reward + reward
    # prev_reward = total_reward  # 現在の報酬を次回のために保存

    return reward

# def calculate_reward(s, prev_s):
#     # 前の状態と比較してq1の速度が増加した場合に報酬を与える
#     if s[2] > prev_s[2]:
#         reward = 10  # 増加した場合の報酬
#     else:
#         reward = 0  # それ以外の場合は報酬を与えない or ペナルティを与える

#     return reward
    # if s[0] > s_prev_q1:
    #     reward = 100
    # else:
    #     reward= -100
    # 速度が速いほど大きな報酬を与える
    # reward = np.abs(s[2]) + np.abs(s[3])  # q1_dot + q2_dot
    # return reward



# Q学習のメイン関数
def q_learning(update_world):
    for epoch in range(15):  
        total_reward = 0
        s = reset()
        q1, q2, q1_dot, q2_dot = s
        q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = discretize_state(s)
        action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)

        # CSVファイルの準備
        csv_file_path = f'dp1_data_epoch_{epoch + 1}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'Reward'])

        # with open(csv_file_path, 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Time', 'Theta', 'Omega', 'Reward'])

            for i in range(6000):
                q1, q2, q1_dot, q2_dot = update_world(i * dt, s, action)
                q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = discretize_state([q1, q2, q1_dot, q2_dot])
                print(f'q1: {q1 * np.pi / 180}, q2: {q2 * np.pi / 180}, q1_dot: {q1_dot},q2_dot: {q2_dot}')

                action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)
                print(action)

                next_q1, next_q2, next_q1_dot, next_q2_dot = update_world(i * dt, [q1, q2, q1_dot, q2_dot], action)
                next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = discretize_state([next_q1, next_q2, next_q1_dot, next_q2_dot])

            # for i in range(6000):
            #     q1, q2, q1_dot, q2_dot = update_world(i * dt, s, action)
            #     q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = discretize_state(s)
            #     print(f'q1: {q1 * np.pi / 180}, q2: {q2 * np.pi / 180}, q1_dot: {q1_dot},q2_dot: {q2_dot}')

            #     action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)
            #     print(action)
            
            #     next_q1, next_q2, next_q1_dot, next_q2_dot = update_world(i * dt, [q1, q2, q1_dot, q2_dot], action)
            #     next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = discretize_state(s)

                # Q値の更新
                reward = calculate_reward([q1, q2, q1_dot, q2_dot], s)  # 報酬を計算

                total_reward += reward
                Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin, action]) + (1 - alpha) * Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

                q1 = next_q1
                q2 = next_q2
                q1_dot = next_q1_dot
                q2_dot = next_q2_dot

                # CSVファイルにデータを保存
                csv_writer.writerow([i * dt, q1, q2, q1_dot, q2_dot, total_reward])

                print(f'Epoch: {epoch + 1}, Total Reward: {total_reward}')
                time.sleep(0.01)

        print(f'Data for epoch {epoch + 1} has been saved to {csv_file_path}')
            

if __name__ == "__main__":
    # mainプログラムの実行
    # main()
    # Q学習の実行
    q_learning(update_world)