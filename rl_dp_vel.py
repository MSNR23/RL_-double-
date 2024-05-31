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
m1 = 10.0
m2 = 10.0

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
        F = np.array([[1.0],[-1.0]])
    elif action == 7:
        F = np.array([[-1.0], [1.0]])
    elif action == 8:
        F = np.array([[0.0], [0.0]])

    # 質量行列
    M_11 = m1*lg1**2 + I1 + m2*(l1**2 + lg2**2 +2*l1*lg2*np.cos(q2))+I2
    M_12 = m2 * (lg2**2 +l1 * lg2*np.cos(q2))+I2
    M_21 = m2 * (lg2**2 + l1*lg2 * np.cos(q2)) + I2
    M_22 = m2 * lg2**2 + I2

    
    M = np.array([[M_11, M_12],
                [M_21, M_22]])

    # コリオリ行列
    C_11 = -m2 * l1 * lg2 * np.sin(q2) * q2_dot * (2 * q1_dot + q2_dot)
    C_21 = m2 * l1 * lg2 * np.sin(q2) * q1_dot**2
    C = np.array([[C_11,], [C_21]])

    # 重力ベクトル
    G_11 = m1 * g * lg1 * np.cos(q1) + m2 * g *(l1 * np.cos(q1) + lg2 * np.cos(q1 + q2))
    G_21 = m2 * g * lg2 * np.cos(q1 + q2)
    G = np.array([[G_11,], [G_21]])

    # 粘性
    B_11 = b1 * q1_dot
    B_21 = b2 * q2_dot
    B = np.array([[B_11,],
                [B_21,]])

    # 逆行列
    M_inv = np.linalg.inv(M)

    # 運動方程式（オイラー法）
    q_dot_dot = M_inv.dot(-C - G + B + F)
    q1_dot_dot = q_dot_dot[0, 0]
    q2_dot_dot = q_dot_dot[1, 0]

    # オイラー法
    q1_new = q1 + q1_dot * dt
    q2_new = q2 + q2_dot * dt
    q1_dot_new = q1_dot + q1_dot_dot * dt
    q2_dot_new = q2_dot + q2_dot_dot * dt

    return (q1_new, q2_new, q1_dot_new, q2_dot_new)


# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.3  # ε-greedy法のε

# Qテーブルの初期化
num_q1_bins = 4
num_q2_bins = 4
num_q1_dot_bins = 4
num_q2_dot_bins = 4
num_actions = 9  # 行動数（例: 0, 1, 2）

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins, num_actions))


def discretize_state(q1, q2, q1_dot, q2_dot):
    q1_bin = np.digitize(q1, np.linspace(-np.pi, np.pi, num_q1_bins + 1)) - 1  
    q2_bin = np.digitize(q2, np.linspace(0, 29 * np.pi / 36, num_q2_bins + 1))  - 1 
    q1_dot_bin = np.digitize(q1_dot, np.linspace(-6.0, 6.0, num_q1_dot_bins + 1)) - 1  
    q2_dot_bin = np.digitize(q2_dot, np.linspace(-6.0, 6.0, num_q2_dot_bins + 1)) - 1  

    # theta_binとomega_binが範囲内に収まるように調整
    q1_bin = np.maximum(0, np.minimum(num_q1_bins - 1, q1_bin))
    q2_bin = np.maximum(0, np.minimum(num_q2_bins - 1, q2_bin))
    q1_dot_bin = np.maximum(0, np.minimum(num_q1_dot_bins - 1, q1_dot_bin))
    q2_dot_bin = np.maximum(0, np.minimum(num_q2_dot_bins - 1, q2_dot_bin))

    return q1_bin, q2_bin, q1_dot_bin, q2_dot_bin

# リセット関数
def reset():
        q1 = q10
        q2 = q20
        q1_dot = q1_dot0
        q2_dot = q2_dot0

        return q1, q2, q1_dot, q2_dot

# ε-greedy法に基づく行動の選択
def select_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, :])


# Q学習のメイン関数
def q_learning(update_world):
    for epoch in range(50):  
        total_reward = 0
        q1, q2, q1_dot, q2_dot = reset()
        q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = discretize_state(q1, q2, q1_dot, q2_dot)
        action = select_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)

        # CSVファイルの準備
        csv_file_path = f'dp2_euler_epoch_{epoch + 1}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'Reward'])


            for i in range(6000):
                q1, q2, q1_dot, q2_dot = update_world(q1, q2, q1_dot, q2_dot, action)
                q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = discretize_state(q1, q2, q1_dot, q2_dot)
                print(f'theta1: {q1}, theta2: {q2}, omega1: {q1_dot}, omega2: {q2_dot}')

                action = select_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin)
                print(action)

                # update_world 関数から必要な値を取得
                next_q1, next_q2, next_q1_dot, next_q2_dot = update_world(q1, q2, q1_dot, q2_dot, action)

                # 取得した値を discretize_state 関数に渡す
                next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = discretize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)

                # Q値の更新
                reward_scale = 0.01
                reward = reward_scale * (q1_dot**2 + abs(q1_dot))
                
                total_reward = reward
                Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin, :]) - Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

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
    # Q学習の実行
    q_learning(update_world)