#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/12/24 16:49
# @Author      :weiz
# @ProjectName :Sfm-python-master
# @File        :sfm.py
# @Description :
import os
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from mayavi import mlab  # 这个安装挺麻烦的，不过不依赖也行，它只是方便展示作用
import time

def extract_feature(image_names):
    """
    extract feature
    :param image_names:
    :return:
    """
    sift = cv2.SIFT_create(0, 3, 0.04, 10)
    key_points_for_all = []  # 关键点信息，包括：angle、class_id、octave、pt、response、size
    descriptor_for_all = []  # 特征点的特征描述符，是一个一维列表，列表元素为Dmatch类型
    colors_for_all = []  # 特征点该点的颜色信息
    image_for_all = []  # 图片

    for image_name in image_names:
        image = cv2.imread(image_name)

        if image is None:
            print("the {} path is error!".format(image_name))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptor = sift.detectAndCompute(gray, None)
        if len(key_points) <= 10:
            print("This {} has less than 10 feature points!".format(image_name))
            continue

        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)

        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image[int(p[1])][int(p[0])]
        colors_for_all.append(colors)

        image_for_all.append(image)

    return np.array(key_points_for_all, dtype=object), np.array(descriptor_for_all, dtype=object),\
           np.array(colors_for_all, dtype=object), image_for_all


def match_features(key_points_for_all, descriptor_for_all, ratio, image_for_all, is_show=None):
    """
    match features
    :param key_points_for_all:
    :param descriptor_for_all:
    :param ratio: Ratio Test
    :param image_for_all:
    :param is_show:
    :return:
    """
    if is_show == True:
        is_show = True
    else:
        is_show = False

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches_for_all = []
    for i in range(len(descriptor_for_all) - 1):
        # 特征匹配可以采用多种方法，下面采用Lowe's算法+KNN近邻算法
        # 可能会出现trainDescCollection[iIdx].rows < IMGIDX_ONE的错误，目前估计是opencv自身的bug
        knn_matches = bf.knnMatch(descriptor_for_all[i], descriptor_for_all[i + 1], k=2)
        good_matches = []
        good_matches_show = []
        # m表示大图像上最匹配点的距离，n表示次匹配点的距离，若比值小于ratio则舍弃
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
                good_matches_show.append([m])

        matches_for_all.append(np.array(good_matches))

        print("Complete the matching of the {}th and {}th pictures!".format(i+1, i+2))
        if is_show:
            img_matches = np.empty((max(image_for_all[i].shape[0], image_for_all[i+1].shape[0]),
                                    image_for_all[i].shape[1] + image_for_all[i+1].shape[1], 3), dtype=np.uint8)
            cv2.drawMatchesKnn(image_for_all[i], key_points_for_all[i], image_for_all[i+1], key_points_for_all[i+1],
                               good_matches_show, img_matches)
            cv2.imshow("matches", img_matches)
            cv2.waitKey(500)

    if is_show:
        cv2.destroyWindow("matches")
    return np.array(matches_for_all, dtype=object)


def get_matched_points(key_points_image1, key_points_image2, matches):
    """
    根据匹配信息获取两张图对应点的坐标信息
    :param key_points_image1:
    :param key_points_image2:
    :param matches:
    :return:
    """
    matche_points_1 = np.asarray([key_points_image1[m.queryIdx].pt for m in matches])
    matche_points_2 = np.asarray([key_points_image2[m.trainIdx].pt for m in matches])

    return matche_points_1, matche_points_2


def get_matched_colors(color_image1, color_image2, matches):
    """
    根据匹配信息获取两张图对应点的颜色信息
    :param color_image1:
    :param color_image2:
    :param matches:
    :return:
    """
    matches_color_1 = np.asarray([color_image1[m.queryIdx] for m in matches])
    matches_color_2 = np.asarray([color_image2[m.trainIdx] for m in matches])

    return matches_color_1, matches_color_2


def get_transform(imp, key_point_1, key_point_2):
    """
    从本质矩阵中恢复R、t,它表示的是key_point_1到key_point_2的变换
    :param imp:
    :param key_point_1:
    :param key_point_2:
    :return:
    """
    E, mask = cv2.findEssentialMat(key_point_1, key_point_2, imp)
    pass_count, R, t, mask = cv2.recoverPose(E, key_point_1, key_point_2, imp, mask=mask)

    return R, t, mask


def mask_points(points, mask):
    """
    选择重合的点
    :param points:
    :param mask:
    :return:
    """
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(points[i])

    return np.array(p1_copy)


def reconstruct(ipm, R1, t1, R2, t2, p1, p2):
    """
    三维重建：特征点的三角化
    :param ipm:
    :param R1:
    :param t1:
    :param R2:
    :param t2:
    :param p1:
    :param p2:
    :return:
    """
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(R1)
    proj1[:, 3] = np.float32(t1.T)
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(t2.T)

    ipm = np.float32(ipm)
    proj1 = np.dot(ipm, proj1)
    proj2 = np.dot(ipm, proj2)
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)

    structure = []
    # 齐次坐标转三维坐标：前三个维度除以第四个维度
    for i in range(len(s[0])):
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])

    return np.array(structure)


def init_sfm(ipm, key_points_for_all, colors_for_all, matches_for_all):
    """
    sfm算法前期准备工作
    :param ipm:Camera internal parameter matrix
    :param key_points_for_all:
    :param colors_for_all:
    :param matches_for_all:
    :return:
    """
    # 获取前两张图匹配到的特征点的坐标以及颜色信息
    matche_p1, matche_p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])
    matche_c1, matche_c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])

    R, t, mask = get_transform(ipm, matche_p1, matche_p2)

    mask_p1 = mask_points(matche_p1, mask)
    mask_p2 = mask_points(matche_p2, mask)
    mask_color = mask_points(matche_c1, mask)
    # print(mask_p1)
    # print(mask_p2)
    # print(mask_color)

    # 以第一张图为参考坐标系,获取图一和图二的三角测量的三维坐标
    R1 = np.eye(3, 3)
    t1 = np.zeros((3, 1))
    structure = reconstruct(ipm, R1, t1, R, t, mask_p1, mask_p2)
    # print(structure)

    rotations = [R1, R]
    motions = [t1, t]
    correspond_struct_idx = []
    for key_p in key_points_for_all:
        # print(np.ones(len(key_p)) * -1)
        correspond_struct_idx.append(np.ones(len(key_p)) * - 1)
    correspond_struct_idx = np.array(correspond_struct_idx, dtype=object)

    idx = 0
    matches = matches_for_all[0]
    for i, match in enumerate(matches):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[1][int(match.trainIdx)] = idx
        idx += 1

    return structure, correspond_struct_idx, mask_color, rotations, motions


def get_3d_points_and_image_points(matches, struct_indices, structure, key_points):
    """
    获取作图像点以及空间点
    :param matches:
    :param struct_indices:
    :param structure:
    :param key_points:
    :return:
    """
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx < 0:
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)

    return np.array(object_points), np.array(image_points)


def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    """
    点云进行融合
    :param matches:
    :param struct_indices:
    :param next_struct_indices:
    :param structure:
    :param next_structure:
    :param colors:
    :param next_colors:
    :return:
    """
    for i, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis=0)
        colors = np.append(colors, [next_colors[i]], axis=0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure, colors


def bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
    """
    光束法平差
    https://blog.csdn.net/OptSolution/article/details/64442962
    :param rotations:
    :param motions:
    :param K:
    :param correspond_struct_idx:
    :param key_points_for_all:
    :param structure:
    :return:
    """
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r
    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3DPose_v2(structure[point3d_id], key_points[j].pt, r, t, K)
            structure[point3d_id] = new_point

    return structure


def get_3DPose_v1(pos, ob, r, t, K):
    """

    :param pos:
    :param ob:
    :param r:
    :param t:
    :param K:
    :return:
    """
    dtype = np.float32
    def F(x):
        p, J = cv2.projectPoints(x.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        err = e
        return err

    res = least_squares(F, pos)
    return res.x


def get_3DPose_v2(pos, ob, r, t, K, x=0.5, y=0.5):
    """

    :param pos:
    :param ob:
    :param r:
    :param t:
    :param K:
    :param x:
    :param y:
    :return:
    """
    p, J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > x or abs(e[1]) > y:
        return None
    return pos


def compute_reprojection_error(structure, rotations, motions, K, correspond_struct_idx, key_points_for_all):
    """
    计算整个重建的重投影误差

    参数:
        structure: 3D点云 (N x 3)
        rotations: 每帧的旋转矩阵列表
        motions: 每帧的平移向量列表
        K: 相机内参矩阵
        correspond_struct_idx: 每帧特征点对应的3D点索引
        key_points_for_all: 每帧的特征点

    返回:
        平均重投影误差（像素）
    """
    total_error = 0
    total_points = 0

    for i in range(len(rotations)):
        # 获取当前帧的位姿
        r_vec, _ = cv2.Rodrigues(rotations[i])
        t_vec = motions[i]

        # 获取当前帧的特征点和对应的3D点索引
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]

        frame_error = 0
        frame_points = 0

        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0 or point3d_id >= len(structure):
                continue

            # 获取3D点和对应的2D观测
            point3d = structure[point3d_id]
            point2d_obs = key_points[j].pt

            # 重投影
            point2d_proj, _ = cv2.projectPoints(
                point3d.reshape(1, 1, 3),
                r_vec, t_vec, K, np.array([])
            )
            point2d_proj = point2d_proj.reshape(2)

            # 计算误差
            error = np.linalg.norm(point2d_obs - point2d_proj)
            frame_error += error
            frame_points += 1

        if frame_points > 0:
            avg_frame_error = frame_error / frame_points
            total_error += frame_error
            total_points += frame_points
            print(f"  第{i + 1}帧重投影误差: {avg_frame_error:.4f}像素 (基于{frame_points}个点)")

    if total_points > 0:
        return total_error / total_points
    else:
        return float('inf')

def show_3D_matplotlib(structure, colors):
    """
    使用matplotlib显示结果
    :param structure:
    :param colors:
    :return:
    """
    colors = colors / 255.0  # 确保是浮点数
    for i in range(len(colors)):
        colors[i, :] = colors[i, :][[2, 1, 0]]  # BGR转RGB

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('3D Reconstruction Result')

    # 方法1：使用add_subplot创建3D坐标轴（推荐）
    ax = fig.add_subplot(111, projection='3d')

    # 或者方法2：使用subplot（等效）
    # ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 绘制点云
    for i in range(len(structure)):
        ax.scatter(structure[i, 0], structure[i, 1], structure[i, 2],
                   color=colors[i, :], s=2)  # s=2 调整点的大小

    # 设置坐标轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 设置视角
    ax.view_init(elev=135, azim=90)

    # 自动调整坐标轴范围
    ax.set_xlim([structure[:, 0].min(), structure[:, 0].max()])
    ax.set_ylim([structure[:, 1].min(), structure[:, 1].max()])
    ax.set_zlim([structure[:, 2].min(), structure[:, 2].max()])

    plt.tight_layout()
    plt.show()

#先不使用mayavi来3D显示
# def show_3D_mayavi_v1(structure):
#     """
#     使用mayavi显示结果
#     :param structure:
#     :return:
#     """
#     mlab.points3d(structure[:, 0], structure[:, 1], structure[:, 2], mode='point', name='dinosaur')
#     mlab.show()
#
#
# def show_3D_mayavi_v2(structure, colors):
#     """
#     使用mayavi显示结果,加上颜色
#     :param structure:
#     :param colors:
#     :return:
#     """
#     colors = colors / 255
#     for i in range(len(structure)):
#         mlab.points3d(structure[i][0], structure[i][1], structure[i][2],
#                       mode='point', name='dinosaur', color=tuple(colors[i]))
#
#     mlab.show()


image_path = "./images2"
ipm = np.array([[2759.48, 0, 1520.69],
                [0, 2764.16, 1006.81],
                [0, 0, 1]])
dpv = np.array([])
ratio = 0.5


def main():
    # 开始计时
    start_time = time.time()

    image_names = os.listdir(image_path)
    image_names = sorted(image_names)

    for i in range(len(image_names)):
        image_names[i] = os.path.join(image_path, image_names[i])

    # 记录各个阶段的耗时
    t1 = time.time()
    key_points_for_all, descriptor_for_all, colors_for_all, image_for_all = extract_feature(image_names)
    t2 = time.time()
    print(f"[耗时] 特征提取: {t2 - t1:.2f}秒")

    matches_for_all = match_features(key_points_for_all, descriptor_for_all, ratio, image_for_all, False)
    t3 = time.time()
    print(f"[耗时] 特征匹配: {t3 - t2:.2f}秒")

    structure, correspond_struct_idx, colors, rotations, motions = init_sfm(ipm, key_points_for_all,
                                                                            colors_for_all, matches_for_all)
    t4 = time.time()
    print(f"[耗时] 初始SFM: {t4 - t3:.2f}秒")

    for i in range(1, len(matches_for_all)):
        object_points, image_points = get_3d_points_and_image_points(matches_for_all[i], correspond_struct_idx[i],
                                                                     structure, key_points_for_all[i + 1])

        if len(image_points) == 0:
            print("The feature points of the {}th picture and the {}th picture do not match!".format(i + 1, i + 2))
            print("It is recommended to delete part of the picture to try or take the picture again!")
            sys.exit()

        retval, R_vector, t, inliers = cv2.solvePnPRansac(object_points, image_points, ipm, dpv)
        R_mat, jacobian = cv2.Rodrigues(R_vector)
        rotations.append(R_mat)
        motions.append(t)

        p1, p2 = get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])
        next_structure = reconstruct(ipm, rotations[i], motions[i], R_mat, t, p1, p2)

        correspond_struct_idx[i], correspond_struct_idx[i + 1], structure, colors = \
            fusion_structure(matches_for_all[i], correspond_struct_idx[i], correspond_struct_idx[i + 1],
                             structure, next_structure, colors, c1)

    t5 = time.time()
    print(f"[耗时] 增量重建: {t5 - t4:.2f}秒")

    # BA优化前记录点数
    points_before_ba = len(structure)
    print(f"[统计] BA优化前点数: {points_before_ba}")

    structure = bundle_adjustment(rotations, motions, ipm, correspond_struct_idx, key_points_for_all, structure)
    t6 = time.time()
    print(f"[耗时] BA优化: {t6 - t5:.2f}秒")

    # 删除无效点
    i = 0
    while i < len(structure):
        if math.isnan(structure[i][0]):
            structure = np.delete(structure, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1

    points_after_ba = len(structure)
    print(f"[统计] BA优化后有效点数: {points_after_ba}")
    print(f"[统计] 移除的无效点数: {points_before_ba - points_after_ba}")

    # 计算重投影误差
    reprojection_error = compute_reprojection_error(
        structure, rotations, motions, ipm,
        correspond_struct_idx, key_points_for_all
    )
    print(f"[精度] 平均重投影误差: {reprojection_error:.4f}像素")

    # 总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print("=" * 50)
    print(f"[总耗时] {total_time:.2f}秒")
    print(f"[总点数] {len(structure)}")
    print(f"[平均误差] {reprojection_error:.4f}像素")
    print("=" * 50)

    np.save('structure.npy', structure)
    np.save('colors.npy', colors)

    # print(structure.shape)
    # print(colors.shape)
    show_3D_matplotlib(structure,colors)
    #show_3D_mayavi_v1(structure)
    # show_3D_mayavi_v2(structure, colors)


if __name__ == "__main__":
    main()