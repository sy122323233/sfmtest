#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件
"""

import cv2
import numpy as np


class Config:
    """全局配置"""

    # 相机内参（根据你的数据集修改）
    K = np.array([
        [2759.48, 0, 1520.69],
        [0, 2764.16, 1006.81],
        [0, 0, 1]
    ])

    # 特征提取配置
    FEATURE = {
        'nfeatures': 0,  # 0表示不限制
        'nOctaveLayers': 3,
        'contrastThreshold': 0.04,
        'edgeThreshold': 10
    }

    # 特征匹配配置
    MATCHING = {
        'ratio': 0.5,  # Lowe's ratio test
        'norm_type': cv2.NORM_L2
    }

    # 重建配置
    RECONSTRUCTION = {
        'min_matches': 8,  # 最小匹配点数
        'reproj_threshold': 0.5  # 重投影误差阈值（像素）
    }

    # BA配置
    BA = {
        'max_iterations': 100,
        'use_cuda': False  # 是否使用CUDA加速
    }

    # 数据集路径
    DATA_PATH = "../images2"

    # 输出路径
    OUTPUT_PATH = "./output"