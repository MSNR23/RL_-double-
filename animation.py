import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# CSVファイルからデータを読み取る関数
def read_csv_data(csv_file_path):
    time_data = []
    theta1_data = []
    theta2_data = []

    with open(csv_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # ヘッダー行をスキップ
        for row in csv_reader:
            time_data.append(float(row[1]))  # Timeデータを読み取る
            theta1_data.append(float(row[2]))  # Theta1データを読み取る
            theta2_data.append(float(row[3]))  # Theta2データを読み取る

    return time_data, theta1_data, theta2_data

# アニメーションの初期化関数
def init():
    pendulum1.set_data([], [])
    pendulum2.set_data([], [])
    return pendulum1, pendulum2

# アニメーションの更新
def update(frame, theta1_data, theta2_data):
    x1 = np.sin(theta1_data[frame])  # 振り子1のx座標
    y1 = -np.cos(theta1_data[frame])  # 振り子1のy座標
    pendulum1.set_data([0, x1], [0, y1])  # 振り子1の位置を更新

    x2 = x1 + np.sin(theta1_data[frame] + theta2_data[frame])  # 振り子2のx座標
    y2 = y1 - np.cos(theta1_data[frame] + theta2_data[frame])  # 振り子2のy座標
    pendulum2.set_data([x1, x2], [y1, y2])  # 振り子2の位置を更新

    return pendulum1, pendulum2

# CSVファイルのパス
csv_file_path = 'data1_episode_1.csv.csv'

# CSVファイルからデータを読み取る
time_data, theta1_data, theta2_data = read_csv_data(csv_file_path)

# アニメーションの設定
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
pendulum1, = ax.plot([], [], lw=2, label='Link 1')
pendulum2, = ax.plot([], [], lw=2, label='Link 2')
ax.legend()

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(time_data), fargs=(theta1_data, theta2_data), init_func=init, blit=False, interval=2)

# アニメーションをMP4形式で保存
ani.save('animation01.mp4', writer='ffmpeg', fps=100)
plt.show()