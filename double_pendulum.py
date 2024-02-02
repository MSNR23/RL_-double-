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
def equations_of_motion(t, s, F):
    q1, q2, q1_dot, q2_dot = s
    
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
    k1 = dt * equations_of_motion(t, s, F)
    k2 = dt * equations_of_motion(t + 0.5 * dt, s + 0.5 * k1, F)
    k3 = dt * equations_of_motion(t + 0.5 * dt, s + 0.5 * k2, F)
    k4 = dt * equations_of_motion(t + dt, s + k3, F)

    s_new = s + (k1 + 2*k2 + 2*k3 + k4) / 6

    return s_new

# アニメーションの作成
def update(frame, line, s_values):
    x1 = l1 * np.cos(s_values[frame, 0])
    y1 = l1 * np.sin(s_values[frame, 0])
    x2 = x1 + l2 * np.cos(s_values[frame, 0] + s_values[frame, 1])
    y2 = y1 + l2 * np.sin(s_values[frame, 0] + s_values[frame, 1])

    line.set_data([0, x1, x2], [0, y1, y2])
    return line,

# main関数
def main():
    # シミュレーションの初期化
    dt = 0.005  # 時間刻み幅
    t_end = 5.0  # シミュレーション終了時間
    t_values = np.arange(0, t_end, dt)
    s_values = np.zeros((len(t_values), 4))

    # 初期条件
    s0 = np.array([np.pi , 0.0, 0.0, 0.0])  # 初期の角度（-90度）、初期の角速度
    F = np.array([0.0, 0.0])  # 外力（ここではゼロ）

    # シミュレーション実行
    for i, t in enumerate(t_values):
        s_values[i] = s0
        s0 = runge_kutta(t, s0, F, dt)  # 修正点: s0を直接更新する

    # データをCSVファイルに保存
    csv_file_path = 'double-pendulum_simulation_data2.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2'])  # ヘッダー行
        for t, theta1, theta2, omega1, omega2 in zip(t_values, s_values[:, 0], s_values[:, 1], s_values[:, 2], s_values[:, 3]):
            csv_writer.writerow([t, theta1, theta2, omega1, omega2])

    print(f'Data has been saved to {csv_file_path}')

    # アニメーションの作成
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)

    ani = FuncAnimation(fig, update, frames=len(t_values), fargs=(line, s_values), interval=50, blit=False)  # intervalを調整

    # アニメーションの保存（GIF形式）
    animation_file_path = 'double-pendulum_animation2.gif'
    ani.save(animation_file_path, writer='pillow', fps=30)
    print(f'Animation has been saved to {animation_file_path}')

    plt.show()

if __name__ == "__main__":
    main()