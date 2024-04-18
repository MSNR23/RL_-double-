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

#振り子の初期角度
theta10 = -np.pi / 6
theta20 = -np.pi / 6
omega10 = 0
omega20 = 0

dt = 0.01


# 運動方程式
def update_world(theta1, theta2, omega1, omega2, action):
    #角度の範囲を制限
    theta1 = np.clip(theta1, -np.pi, np.pi)
    theta2 = np.clip(theta2, -np.pi, np.pi)
    if action == 0:
        tau1 = 0
        tau2 = 0
    elif action == 1:
        tau1 = 1
        tau2 = 0
    elif action == 2:
        tau1 = 0
        tau2 = 1
    elif action == 3:
        tau1 = -1
        tau2 = 0
    elif action == 4:
        tau1 = 0
        tau2 = -1
    elif action == 5:
        tau1 = 1
        tau2 = 1
    elif action == 6:
        tau1 = 1
        tau2 = -1
    elif action == 7:
        tau1 = -1
        tau2 = 1
    elif action == 8:
        tau1 = -1
        tau2 = -1
    
    # 質量行列
    M = np.array([[m1*lg1**2 + I1 + m2*(l1**2 + lg2**2 +2*l1*lg2*np.cos(theta2))+I2, m2 * (lg2**2 +l1 * lg2*np.cos(theta2))+I2],
                  [m2 * (lg2**2 + l1*lg2 * np.cos(theta2)) + I2, m2 * lg2**2 + I2]])

    # コリオリ行列
    C = np.array([[-m2 * l1 * lg2 * np.sin(theta2) * omega2 * (2 * omega1 + omega2)],
                  [m2 * l1 * lg2 * np.sin(theta2) * omega1**2]])

    # 重力ベクトル
    G = np.array([[m1 * g * lg1 * np.cos(theta1) + m2 * g *(l1 * np.cos(theta1) + lg2 * np.cos(theta1 + theta2))],
               [m2 * g * lg2 * np.cos(theta1 + theta2)]])

    # 粘性
    B = np.array([[b1 * omega1],
                  [b2 * omega2]])

    # 外力
    # tau = np.array([[tau[0], 0],
    #               [0, tau[1]]], dtype=float)
    tau = np.array([[tau1], [tau2]])

    # 逆行列
    M_inv = np.linalg.inv(M)

    q_ddot = M_inv.dot(-C - G + B + tau)

    return  np.array([omega1, omega2, q_ddot[0, 0], q_ddot[1, 0]])

# Runge-Kutta法
def runge_kutta(t, theta1, theta2, omega1, omega2, tau, dt):
    k1 = dt * update_world(t, [theta1, theta2, omega1, omega2], tau)
    k2 = dt * update_world(t + 0.5 * dt, [theta1 + 0.5 * k1[0], theta2 + 0.5 * k1[1], omega1 + 0.5 * k1[2], omega2 + 0.5 * k1[3]], tau)
    k3 = dt * update_world(t + 0.5 * dt, [theta1 + 0.5 * k2[0], theta2 + 0.5 * k2[1], omega1 + 0.5 * k2[2], omega2 + 0.5 * k2[3]], tau)
    k4 = dt * update_world(t + dt, [theta1 + k3[0], theta2 + k3[1], omega1 + k3[2], omega2 + k3[3], tau])

    theta1_new = theta1 + (k1[0] + 2 * k2[0] + 2 * k3[0]  + k4[0]) / 6
    theta2_new = theta2 + (k1[1] + 2 * k2[1] + 2 * k3[1]  + k4[1]) / 6
    omega1_new = omega1 + (k1[2] + 2 * k2[2] + 2 * k3[2]  + k4[2]) / 6
    omega2_new = omega2 + (k1[3] + 2 * k2[3] + 2 * k3[3]  + k4[3]) / 6

    return theta1_new, theta2_new, omega1_new, omega2_new


# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.3  # ε-greedy法のε

# Qテーブルの初期化
num_theta1_bins = 4
num_theta2_bins = 4
num_omega1_bins = 4
num_omega2_bins = 4
num_actions = 9  # 行動数（例: 0, 1, 2）

Q = np.zeros((num_theta1_bins, num_theta2_bins, num_omega1_bins, num_omega2_bins, num_actions))

# 状態の離散化関数
def discretize_state(theta1, theta2, omega1, omega2):
    theta1_bin = np.digitize(theta1, np.linspace(-np.pi, np.pi, num_theta1_bins + 1)) - 1
    theta2_bin = np.digitize(theta2, np.linspace(0, 29 * np.pi / 36, num_theta2_bins + 1)) - 1

    omega1_bin = np.digitize(omega1, np.linspace(-2.0, 2.0, num_omega1_bins + 1)) - 1
    omega2_bin = np.digitize(omega2, np.linspace(-2.0, 2.0, num_omega2_bins + 1)) - 1

    # theta_binとomega_binが範囲内に収まるように調整
    theta1_bin = max(0, min(num_theta1_bins - 1, theta1_bin))
    theta2_bin = max(0, min(num_theta2_bins - 1, theta2_bin))

    omega1_bin = max(0, min(num_omega1_bins - 1, omega1_bin))
    omega2_bin = max(0, min(num_omega2_bins - 1, omega2_bin))

    return theta1_bin, theta2_bin, omega1_bin, omega2_bin

# リセット関数
def reset():
    theta1 = theta10
    theta2 = theta20
    omega1 = omega10
    omega2 = omega20

    return theta1, theta2, omega1, omega2

# ε-greedy法に基づく行動の選択
def select_action(theta1_bin, theta2_bin, omega1_bin, omega2_bin):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[theta1_bin, theta2_bin, omega1_bin, omega2_bin, :])


# Q学習のメイン関数
def q_learning(update_world):

    for epoch in range(5):  
        total_reward = 0
        theta1, theta2, omega1, omega2 = reset()
        theta1_bin, theta2_bin, omega1_bin, omega2_bin = discretize_state(theta1, theta2, omega1, omega2)
        action = select_action(theta1_bin, theta2_bin, omega1_bin, omega2_bin)

        # CSVファイルの準備
        csv_file_path = f'rl_test_data_epoch_{epoch + 1}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Time', 'Theta', 'Omega', 'Reward'])


            for i in range(6000):
                print("theta1_bin:", theta1_bin)
                print("theta2_bin:", theta2_bin)
                print("omega1_bin:", omega1_bin)
                print("omega2_bin:", omega2_bin)
                print("action:", action)
                next_theta1, next_theta2, next_omega1, next_omega2 = update_world(theta1, theta2, omega1, omega2, action)
                theta1_bin, theta2_bin, omega1_bin, omega2_bin = discretize_state(theta1, theta2, omega1, omega2)
                print(f'theta1: {theta1 * np.pi / 180}, theta2: {theta2 * np.pi / 180}, omega1: {omega1}, omega2: {omega2}')

                action = select_action(theta1_bin, theta2_bin, omega1_bin, omega2_bin)
                print(action)
            
                next_theta1, next_theta2, next_omega1, next_omega2 = update_world(theta1, theta2, omega1, omega2, action)
                theta1_bin, theta2_bin, omega1_bin, omega2_bin = discretize_state(next_theta1, next_theta2, next_omega1, next_omega2)
                
                # CSVファイルにデータを保存
                # csv_writer.writerow([i * dt, theta, omega])

                # Q値の更新
                # reward_scale = 0.01
                # reward = reward_scale * (theta1**2 +  omega1**2 + 0.1 * (next_theta1 - theta1)**2 + 0.1 * (next_omega1 - omega1)**2) 
                # 新しい報酬の計算
                # reward_scale = 0.01
                # # 回転速度の大きさを報酬に反映する
                # reward = reward_scale * (theta1**2 +  omega1**2 + 0.1 * (next_theta1 - theta1)**2 + 0.1 * (next_omega1 - omega1)**2 + 0.5 * (omega1**2 + omega2**2))
                reward_scale = 0.01
                # 回転速度の大きさを報酬に反映する
                reward = reward_scale * (theta1**2 +  omega1**2 + 0.1 * (next_theta1 - theta1)**2 + 0.1 * (next_omega1 - omega1)**2 + 0.5 * (omega1**2 + omega2**2 + next_omega1**2 + next_omega2**2))


                # if theta < -np.pi / 2 or theta > np.pi / 2:
                #     reward +=  - 500

                total_reward = reward
                # Q[theta1_bin, theta2_bin, omega1_bin, omega2_bin, action] += alpha * (reward + gamma * np.max(Q[next_theta1, next_theta2, next_omega1, next_omega2, action]) + (1 - alpha) * Q[theta1_bin, theta2_bin, omega1_bin, omega2_bin, action])
                # Q[int(theta1_bin), int(theta2_bin), int(omega1_bin), int(omega2_bin), int(action)] += alpha * (reward + gamma * np.max(Q[int(next_theta1), int(next_theta2), int(next_omega1), int(next_omega2), int(action)]) + (1 - alpha) * Q[int(theta1_bin), int(theta2_bin), int(omega1_bin), int(omega2_bin), int(action)])
                next_theta1_bin, next_theta2_bin, next_omega1_bin, next_omega2_bin = discretize_state(next_theta1, next_theta2, next_omega1, next_omega2)
                Q[int(theta1_bin), int(theta2_bin), int(omega1_bin), int(omega2_bin), int(action)] += alpha * (reward + gamma * np.max(Q[next_theta1_bin, next_theta2_bin, next_omega1_bin, next_omega2_bin, int(action)]) + (1 - alpha) * Q[int(theta1_bin), int(theta2_bin), int(omega1_bin), int(omega2_bin), int(action)])


                theta1 = next_theta1
                theta2 = next_theta2
                omega1 = next_omega1
                omega2 = next_omega2

                # CSVファイルにデータを保存
                csv_writer.writerow([i * dt, theta1, theta2, omega1, omega2, total_reward])



                print(f'Epoch: {epoch + 1}, Total Reward: {total_reward}')
                time.sleep(0.01)

        print(f'Data for epoch {epoch + 1} has been saved to {csv_file_path}')
            

if __name__ == "__main__":
    # mainプログラムの実行
    # main()
    # Q学習の実行
    q_learning(update_world)