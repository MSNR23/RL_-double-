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
save_dir = r'data2'

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

     # 角度制約
    q2_new = np.clip(q2_new, 0, 29 * np.pi / 36)
    return q1_new, q2_new, q1_dot_new, q2_dot_new


# Q学習のパラメータ
alpha = 0.05  # 学習率
gamma = 0.95  # 割引率
initial_epsilon = 0.3  # 初期探索率
min_epsilon = 0.01  # 最小探索率
epsilon_decay = 0.999  # 探索率の減少係数


# Qテーブルの初期化
num_q1_bins = 30
num_q2_bins = 30
num_q1_dot_bins = 30
num_q2_dot_bins = 30
num_actions = 9  # 行動数

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins, num_actions))


def discretize_state(q1, q2, q1_dot, q2_dot):
    q1_bin = np.digitize(q1, np.linspace(-np.pi, np.pi, num_q1_bins + 1)) - 1  
    q2_bin = np.digitize(q2, np.linspace(0, 29 * np.pi / 36, num_q2_bins + 1))  - 1 
    q1_dot_bin = np.digitize(q1_dot, np.linspace(-6.0, 6.0, num_q1_dot_bins + 1)) - 1  
    q2_dot_bin = np.digitize(q2_dot, np.linspace(-6.0, 6.0, num_q2_dot_bins + 1)) - 1  

    q1_bin = np.clip(q1_bin, 0, num_q1_bins - 1)
    q2_bin = np.clip(q2_bin, 0, num_q2_bins - 1)
    q1_dot_bin = np.clip(q1_dot_bin, 0, num_q1_dot_bins - 1)
    q2_dot_bin = np.clip(q2_dot_bin, 0, num_q2_dot_bins - 1)

    return q1_bin, q2_bin, q1_dot_bin, q2_dot_bin

# リセット関数
def reset():
    q1 = -np.pi / 8  # 初期角度 q1
    q2 = 0  # 初期角度 q2
    q1_dot = 0.0  # 初期角速度 q1_dot
    q2_dot = 0.0  # 初期角速度 q2_dot
    return q1, q2, q1_dot, q2_dot



# ε-greedy法に基づく行動の選択
def select_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, :])

def calculate_reward(q1, q1_dot, initial_q1, previous_q1, completed_full_rotation):
    reward = 0
    if q1 > previous_q1:
        reward = 2
    else:
        reward = -1

    # 1周して起点に戻った場合の追加報酬
    if completed_full_rotation and q1 < initial_q1:
        reward += 1000

    return reward


# Q学習のメイン関数
def q_learning(update_world):
    epsilon = initial_epsilon
    for epoch in range(30):
        total_reward = 0
        q1, q2, q1_dot, q2_dot = reset()
        initial_q1 = q1
        previous_q1 = q1
        completed_full_rotation = False

        q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = discretize_state(q1, q2, q1_dot, q2_dot)
        action = select_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, epsilon)

        csv_file_path = os.path.join(save_dir, f'data1_epoch_{epoch + 1}.csv.csv')
        with open(csv_file_path, 'a', newline='') as csvfile:  # 'w'モードで追記
            csv_writer = csv.writer(csvfile)
            if os.stat(csv_file_path).st_size == 0:  # ファイルが空の場合、ヘッダーを書き込む
                csv_writer.writerow(['Epoch', 'Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'Reward'])

            for i in range(6000):
                next_q1, next_q2, next_q1_dot, next_q2_dot = update_world(q1, q2, q1_dot, q2_dot, action)
                next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = discretize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)

                # 1周したかどうかの判定
                if next_q1 >= initial_q1 + 2 * np.pi:
                    completed_full_rotation = True

                reward = calculate_reward(next_q1, next_q1_dot, initial_q1, previous_q1, completed_full_rotation)
                previous_q1 = next_q1

                total_reward += reward

                # Q値の更新
                Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin, :]) - Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

                # 状態の更新
                q1, q2, q1_dot, q2_dot = next_q1, next_q2, next_q1_dot, next_q2_dot
                q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin
                action = select_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, epsilon)

                csv_writer.writerow([epoch + 1, i * dt, q1, q2, q1_dot, q2_dot, total_reward])

                print(f'Epoch: {epoch + 1}, Step: {i}, Total Reward: {total_reward}, Q1: {q1}, Q2: {q2}, Q1_dot: {q1_dot}, Q2_dot: {q2_dot}, Action: {action}')

                time.sleep(0.01)

                # 1周して起点に戻った場合にエピソードを終了
                if completed_full_rotation and q1 < initial_q1:
                    print(f"Completed a full rotation and returned to the starting point in epoch {epoch + 1} at step {i + 1}. Ending episode.")
                    break

        # エポックごとに探索率を減少させる
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f'Epsilon after epoch {epoch + 1}: {epsilon}')
        print(f'Total reward for epoch {epoch + 1}: {total_reward}')

        print(f'Data for epoch {epoch + 1} has been saved to {csv_file_path}')

            
if __name__ == "__main__":
    q_learning(update_world)

