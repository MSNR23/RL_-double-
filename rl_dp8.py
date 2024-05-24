import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

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

    return np.array([q1_dot, q2_dot, q_ddot[0, 0], q_ddot[1, 0]])

# Runge-Kutta法
def runge_kutta(t, s, F, dt):
    k1 = dt * update_world(t, s, F)
    k2 = dt * update_world(t + 0.5 * dt, s + 0.5 * k1, F)
    k3 = dt * update_world(t + 0.5 * dt, s + 0.5 * k2, F)
    k4 = dt * update_world(t + dt, s + k3, F)

    s_new = s + (k1 + 2*k2 + 2*k3 + k4) / 6

    return s_new

class QLearning:
    def __init__(self, num_states, num_actions, num_bins, lr=0.1, gamma=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.lr = lr  # 学習率
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # 探索率
        self.q_table = np.zeros((np.prod(num_bins), num_actions))
        # print("Qテーブルのサイズ:", self.q_table.shape)  # 追加: Qテーブルのサイズを出力
        self.num_bins_cumulative = np.concatenate(([1], np.cumprod(num_bins[:-1])))
        self.num_bins = num_bins
        self.episode_count = 0 # エピソード数のカウント
        self.learning_steps = 0 # 学習回数のカウント


    def get_action(self, state):
        # ε-greedy法
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            state_index = self.state_to_index(state)  # 状態をインデックスに変換
            if state_index >= len(self.q_table):  # 状態インデックスがq_tableの範囲外の場合
                state_index = len(self.q_table) - 1  # 最大インデックスに修正
        return np.argmax(self.q_table[state_index, :])
    

    def state_to_index(self, state):
        index = 0
        for i in range(len(state)):
            # 状態の値を適切なビンにマッピング
            state_value = min(max(state[i], 0), self.num_bins[i] - 1)
            # ビンのインデックスを計算
            index += state_value * self.num_bins_cumulative[i]
            # インデックスがQテーブルのサイズを超えないように調整
            index = min(index, len(self.q_table) - 1)
        return index



    def update_q_table(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        # print("State index:", state_index)
        next_state_index = self.state_to_index(next_state)
        predict = self.q_table[state_index, action]
        target = reward + self.gamma * np.max(self.q_table[next_state_index])
        self.q_table[state_index, action] += self.lr * (target - predict)

        # 追加：エピソード数と学習回数を更新
        self.episode_count += 1
        self.learning_steps += 1

        # # エピソード数と学習回数を表示
        # if self.episode_count % 100 == 0:
        #     print(f"エピソード数: {self.episode_count}, 学習回数: {self.learning_steps}")



# 振り子の状態を取得する関数
def digitize_state(s):
    q1_bins = np.linspace(-np.pi, np.pi, 11)  # q1 のビン境界
    q2_bins = np.linspace(0, 29 * np.pi / 36, 10)  # q2 のビン境界
    q1_dot_bins = np.linspace(-6.0, 6.0, 12)  # q1_dot のビン境界
    q2_dot_bins = np.linspace(-6.0, 6.0, 12)  # q2_dot のビン境界

    state = [
        np.digitize(s[0], q1_bins) - 1,  # np.digitize は1から始まるので調整
        np.digitize(s[1], q2_bins) - 1,
        np.digitize(s[2], q1_dot_bins) - 1,
        np.digitize(s[3], q2_dot_bins) - 1
    ]

    return tuple(state)


# 報酬を計算する関数
def calculate_reward(s, s_prev_q1):
    if s[0] > s_prev_q1:
        reward = 100
    else:
        reward= -100
    # # 速度が速いほど大きな報酬を与える
    # reward = np.abs(s[2]) + np.abs(s[3])  # q1_dot + q2_dot
    return reward

def main():
    # シミュレーションの初期化
    dt = 0.01  # 時間刻み幅
    t_end = 60.0  # シミュレーション終了時間
    num_steps = int(t_end / dt)  # ステップ数
    t_values = np.linspace(0, t_end, num_steps)  # 時間配列

    # Q学習の初期化
    num_states = 4  # 状態空間の次元
    num_actions = 2  # 行動空間の次元
    num_bins = [10, 10, 10, 10]
    q_learning = QLearning(num_states, num_actions, num_bins)

    for episode in range(1000):
        # シミュレーションの初期化
        s_values = np.zeros((num_steps, 4))  # ここで初期化
        reward_values = np.zeros(num_steps)  # 報酬の初期化
        s_prev_q1 = -np.pi # 前ステップのq1を保持

        # 初期条件
        s0 = np.array([-2*np.pi/3, -2*np.pi/3, 0.0, 0.0])  # 初期の角度（-2π/3）、初期の角速度
        F = np.array([0.0, 0.0])  # 外力（ここではゼロ）

        # シミュレーション実行
        for i, t in enumerate(t_values):
            s_values[i] = s0
            state = digitize_state(s0)  # 状態を取得
            action = q_learning.get_action(state)  # 行動を選択
            s0 = runge_kutta(t, s0, action, dt)  # 行動を反映してシミュレーションを実行
            next_state = digitize_state(s0)  # 次の状態を取得
            reward = calculate_reward(s0, s_prev_q1)  # 報酬を計算
            reward_values[i] = reward  # 報酬を保存
            q_learning.update_q_table(state, action, reward, next_state)  # Q値を更新

            # 学習終了条件
            if i == num_steps -1 or s0[0] >= np.pi:
                break

            s_prev_q1 = s0[0] # Q値の更新
            
            # エピソード数と各ステップの状態と報酬を表示
            print(f"エピソード: {episode + 1}, ステップ: {i + 1}, q1: {s0[0]}, q2: {s0[1]}, q1_dot: {s0[2]}, q2_dot: {s0[3]}, 報酬: {reward}")


        # データをCSVファイルに保存
        csv_file_path = f'double-pendulum_simulation_data_epoch_{episode + 1}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'Reward'])  # ヘッダー行
            for t, theta1, theta2, omega1, omega2, reward in zip(t_values, s_values[:, 0], s_values[:, 1], s_values[:, 2], s_values[:, 3], reward_values):
                csv_writer.writerow([t, theta1, theta2, omega1, omega2, reward])

        print(f'Data for epoch {episode + 1} has been saved to {csv_file_path}')

if __name__ == "__main__":
    main()
