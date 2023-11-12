#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2023/11/12 20:54:06
@Author : LuZhanglin

相机位姿绘图
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_camera(ax, R, t):
    # 绘制相机坐标轴
    ax.quiver(t[0], t[1], t[2], R[0, 0], R[0, 1], R[0, 2], color='r', label='X')
    ax.quiver(t[0], t[1], t[2], R[1, 0], R[1, 1], R[1, 2], color='g', label='Y')
    ax.quiver(t[0], t[1], t[2], R[2, 0], R[2, 1], R[2, 2], color='b', label='Z')


if __name__ == "__main__":
    pose_dir = Path(__file__).absolute().parent / "data/nerf_llff_data/fern/poses_bounds.npy"
    poses_arr = np.load(pose_dir)  # shape[1] = 17 = 3*4+3 + 2= [R T | [H W F]^T], 最后两个是near far的值
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # 变成[R T | [H W F]^T]矩阵， shape [num of poses, 3,5]

    # 创建一个3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # 画相机
    for pose_i in poses:
        # 相机的位姿
        R = pose_i[:, :3]
        t = pose_i[:, 3]
        draw_camera(ax, R, t)

    # 设置坐标轴范围
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()
