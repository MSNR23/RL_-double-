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
m1 = 5.0
m2 = 5.0

# 慣性モーメント
I1 = m1 * l1**2 / 3
I2 = m2 * l2**2 / 3

# 粘性係数
b1 = 0.1
b2 = 0.1

# 初期条件
q10 = -3 * np.pi / 2
q20 = np.pi / 4
q1_dot0 = 0.0
q2_dot0 = 0.0

dt = 0.001

# CSVファイルの保存先ディレクトリ
save_dir = r'try7'

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 運動方程式（オイラー法）
def update_world(q1, q2, q1_dot, q2_dot, F, action):
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

    # リンク2が可動範囲の限界に達した場合の外力
    if q2 <= 0:
        F[1, 0] = 10.0  # 0度のとき、正の方向に5N
    elif q2 >= np.radians(145):
        F[1, 0] = -10.0  # 145度のとき、負の方向に5N

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

    q_ddot = M_inv.dot(F - C - G - B)


    return np.array([q1_dot, q2_dot, q_ddot[0, 0], q_ddot[1, 0]])

# Runge-Kutta法
def runge_kutta(t, q1, q2, q1_dot, q2_dot, action, dt):
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

    k1 = dt * update_world(q1, q2, q1_dot, q2_dot, F, action)
    k2 = dt * update_world(q1 + 0.5 * k1[0], q2 + 0.5 * k1[1], q1_dot + 0.5 * k1[2], q2_dot + 0.5 * k1[3], F, action)
    k3 = dt * update_world(q1 + 0.5 * k2[0], q2 + 0.5 * k2[1], q1_dot + 0.5 * k2[2], q2_dot + 0.5 * k2[3], F, action)
    k4 = dt * update_world(q1 + k3[0], q2 + k3[1], q1_dot + k3[2], q2_dot + k3[3], F, action)

    q1_new = q1 + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    q2_new = q2 + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    q1_dot_new = q1_dot + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    q2_dot_new = q2_dot + (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6

    # # リンク2の角度を0~145度に制限
    # if q2_new < 0:
    #     q2_new = 0
    #     q2_dot_new = max(q2_dot_new, 0)
    # elif q2_new > np.radians(145):
    #     q2_new = np.radians(145)
    #     q2_dot_new = min(q2_dot_new, 0)


    # # リンク2の角度を0~145度に制限
    # q2_new = max(0, min(np.radians(145), q2_new))
     # 角度制約を追加
    # q2_new = np.clip(q2_new, 215 * np.pi / 180, 359 * np.pi / 180)

    q2_new = np.clip(q2_new, 0, 145 * np.pi / 180)

    return q1_new, q2_new, q1_dot_new, q2_dot_new

max_number_of_steps = 6000 # 最大ステップ数
num_episodes = 300

# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.3  # ε-greedy法のε

# Q = np.random.uniform(low = 1, high = 1, size = (30**30, 9))
# Qテーブルの初期化
num_q1_bins = 4
num_q2_bins = 4
num_q1_dot_bins = 4
num_q2_dot_bins = 4
num_actions = 9  # 行動数

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins, num_actions))

# # 状態の離散化関数
# def digitize_state(q1, q2, q1_dot, q2_dot):
#     q1_bin = np.digitize(q1, np.linspace(-np.pi, np.pi, num_q1_bins + 1)) - 1
#     q2_bin = np.digitize(q2, np.linspace(-np.pi, np.pi, num_q2_bins + 1)) - 1

#     q1_dot_bin = np.digitize(q1_dot, np.linspace(-4.0, 4.0, num_q1_dot_bins + 1)) - 1
#     q2_dot_bin = np.digitize(q2_dot, np.linspace(-4.0, 4.0, num_q2_dot_bins + 1)) - 1

#     # theta_binとomega_binが範囲内に収まるように調整
#     q1_bin = max(0, min(num_q1_bins - 1, q1_bin))
#     q2_bin = max(0, min(num_q2_bins - 1, q2_bin))
#     q1_dot_bin = max(0, min(num_q1_dot_bins - 1, q1_dot_bin))
#     q2_dot_bin = max(0, min(num_q2_dot_bins - 1, q2_dot_bin))

#     return q1_bin, q2_bin, q1_dot_bin, q2_dot_bin
# 状態の離散化関数
def digitize_state(q1, q2, q1_dot, q2_dot):
    q1_bins = np.linspace(-np.pi, np.pi, num_q1_bins + 1)
    q2_bins = np.linspace(215 * np.pi / 180, 359 * np.pi / 180, num_q2_bins + 1)  # リンク2の範囲を0~145度に設定
    q1_dot_bins = np.linspace(-10.0, 10.0, num_q1_dot_bins + 1)
    q2_dot_bins = np.linspace(-10.0, 10.0, num_q2_dot_bins + 1)

    q1_bin = np.digitize(q1, q1_bins) - 1
    q2_bin = np.digitize(q2, q2_bins) - 1
    q1_dot_bin = np.digitize(q1_dot, q1_dot_bins) - 1
    q2_dot_bin = np.digitize(q2_dot, q2_dot_bins) - 1

    # ビンが範囲外の場合の処理
    q1_bin = max(0, min(num_q1_bins - 1, q1_bin))
    q2_bin = max(0, min(num_q2_bins - 1, q2_bin))
    q1_dot_bin = max(0, min(num_q1_dot_bins - 1, q1_dot_bin))
    q2_dot_bin = max(0, min(num_q2_dot_bins - 1, q2_dot_bin))

    return q1_bin, q2_bin, q1_dot_bin, q2_dot_bin

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

# 報酬関数の修正
def compute_reward(q1, q2, q1_dot, q2_dot, next_q1, next_q2):
    reward = q1 + 10 * q1_dot  # リンク1の角速度に応じた報酬 

    # if 
    
    return reward

# Q学習のメイン関数
def q_learning(runge_kutta):
    for epoch in range(num_episodes):  
        total_reward = 0
        sumReward = 0
        q1, q2, q1_dot, q2_dot = reset()
        q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
        action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)

        # CSVファイルの準備
        csv_file_path = os.path.join(save_dir, f'try_{epoch + 1}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Time', 'Theta1', 'Theta2','Omega1','Omega2', 'Reward'])


            for i in range(max_number_of_steps):
                q1, q2, q1_dot, q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, action, dt)
                # リンク2の角度を0~145度に制限
                # q2 = max(0, min(np.radians(145), q2))

                q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
                print(f'theta1: {q1 * 180 / np.pi}, theta2: {q2 * 180 / np.pi}, omega1: {q1_dot}, omega2: {q2_dot}')

                next_action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)
                print(action)
                next_q1, next_q2, next_q1_dot, next_q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, next_action, dt)            

                # リンク2の角度を0~145度に制限
                # next_q2 = max(0, min(np.radians(145), next_q2))
       
                next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = digitize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)

                # # Q値の更新
                # reward_scale = 1
                # reward = reward_scale * ((next_q1 - q1) +(next_q1_dot - q1_dot))  
                # if next_q1 > q1:
                #     reward += 1
                # else:
                #     reward += -1

                # total_reward += reward

                # 修正された報酬の計算
                reward = compute_reward(q1, q2, q1_dot, q2_dot, next_q1, next_q2)
                total_reward += reward
                sumReward += gamma ** (i + 1) * reward
                Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin, action]) + (1 - alpha) * Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

                q1 = next_q1
                q2 = next_q2
                q1_dot = next_q1_dot
                q2_dot = next_q2_dot
                action = next_action

                # CSVファイルにデータを保存
                csv_writer.writerow([i * dt, q1, q2, q1_dot, q2_dot, total_reward])

                print(f'Epoch: {epoch + 1}, Step: {i}, Total Reward: {total_reward}')
                time.sleep(0.01)

        print(f'Data for epoch {epoch + 1} has been saved to {csv_file_path}')
            

if __name__ == "__main__":
    # mainプログラムの実行
    # main()
    # Q学習の実行
    q_learning(runge_kutta)