#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2023/11/11 17:08:27
@Author : LuZhanglin

1) 推导并写出对给定空间中的三维点(𝑥, 𝑦, 𝑧),由 RDF 坐标系转换为 DRB 坐标
系的旋转矩阵，写出详细步骤和计算过程；
2) 推导并写出对给定空间中的三维点(𝑥, 𝑦, 𝑧),由 DRB 坐标系转换为 RUB 坐标
系的旋转矩阵，写出详细步骤和计算过程；
3) 推导并写出对给定空间中的三维点(𝑥, 𝑦, 𝑧),由 RDF 坐标系转换为 RUB 坐标
系的旋转矩阵，写出详细步骤和计算过程。
4) 编程实现：
'''

import numpy as np


def rdf2drb(xyz):
    transform_matrix = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, -1]])
    return transform_matrix @ xyz


def drb2rub(xyz):
    transform_matrix = np.array([[0, 1, 0],
                                 [-1, 0, 0],
                                 [0, 0, 1]])
    return transform_matrix @ xyz


def rdf2rub(xyz):
    transform_matrix = np.array([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]])
    return transform_matrix @ xyz


if __name__ == "__main__":
    # each row is xyz of a point
    xyz = np.array([[1, 2, 3],
                    [4, 5, 6]])
    print(rdf2drb(xyz.T))
    print(drb2rub(xyz.T))
    print(rdf2rub(xyz.T))
