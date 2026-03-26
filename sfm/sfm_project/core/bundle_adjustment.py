#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光束法平差模块 - 过滤版本（匹配原始代码）
"""

import cv2
import numpy as np


class BundleAdjustment:
    """光束法平差 - 过滤异常点"""

    def __init__(self, K):
        self.K = K

    def optimize(self, points_3d, rotations, motions, key_points_list, correspondences):
        """
        过滤异常点（不改变点云位置）

        原始代码的策略：
        1. 检查每个3D点的重投影误差
        2. 误差大于阈值的点被标记
        3. 只保留误差小的点
        """
        print("  开始过滤异常点...")

        # 复制点云
        filtered_points = points_3d.copy()

        # 记录哪些点要保留
        keep_mask = np.ones(len(points_3d), dtype=bool)

        # 统计每个点的误差
        point_errors = np.zeros(len(points_3d))
        point_count = np.zeros(len(points_3d))

        # 对每个相机
        for i in range(len(rotations)):
            if i >= len(key_points_list):
                continue

            # 转换为旋转向量（原始代码的做法）
            r_vec, _ = cv2.Rodrigues(rotations[i])
            t_vec = motions[i]

            key_points = key_points_list[i]
            corr = correspondences[i] if i < len(correspondences) else []

            # 对每个特征点
            for j in range(min(len(corr), len(key_points))):
                point3d_id = int(corr[j])
                if point3d_id < 0 or point3d_id >= len(points_3d):
                    continue

                # 获取3D点和观测点
                point3d = points_3d[point3d_id]
                point2d_obs = key_points[j].pt

                # 重投影
                point2d_proj, _ = cv2.projectPoints(
                    point3d.reshape(1, 1, 3),
                    r_vec, t_vec, self.K, np.array([])
                )
                point2d_proj = point2d_proj.reshape(2)

                # 计算误差
                error = np.linalg.norm(point2d_obs - point2d_proj)
                point_errors[point3d_id] += error
                point_count[point3d_id] += 1

        # 计算平均误差并过滤（使用与原始代码相同的阈值 x=0.5, y=0.5）
        # 原始代码中 get_3DPose_v2 的阈值是 x=0.5, y=0.5 像素
        reproj_threshold = 0.5  # 与原始代码一致

        for i in range(len(points_3d)):
            if point_count[i] > 0:
                avg_error = point_errors[i] / point_count[i]
                if avg_error > reproj_threshold:
                    keep_mask[i] = False
            else:
                keep_mask[i] = False

        # 应用过滤
        filtered_points = points_3d[keep_mask]

        removed_count = len(points_3d) - len(filtered_points)
        print(f"  过滤前点数: {len(points_3d)}")
        print(f"  过滤后点数: {len(filtered_points)}")
        print(f"  移除点数: {removed_count} ({removed_count / len(points_3d) * 100:.1f}%)")
        print("  BA过滤完成")

        return filtered_points

    def optimize_cuda(self, points_3d, rotations, motions, key_points_list, correspondences):
        """CUDA版本（暂未实现）"""
        print("  CUDA版本BA（使用CPU版本代替）")
        return self.optimize(points_3d, rotations, motions, key_points_list, correspondences)