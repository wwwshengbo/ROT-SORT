import numpy as np

# 为下面的 m_distance 准备
def diff_orientation_correction(diff):
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    return diff


# def m_distance(det, trk, trk_inv_innovation_matrix=None):
#     # 将检测框和跟踪框转换为数组，只取前7个元素（可能表示边界框的位置和尺寸信息）
#     det_array = BBox.bbox2array(det)[:7]
#     trk_array = BBox.bbox2array(trk)[:7]
#     # 计算检测框和跟踪框之间的差异向量
#     diff = np.expand_dims(det_array - trk_array, axis=1)
#     # 对偏航角差异进行修正
#     corrected_yaw_diff = diff_orientation_correction(diff[3])
#     diff[3] = corrected_yaw_diff
#
#     # 如果提供了逆创新矩阵，则计算马氏距离
#     if trk_inv_innovation_matrix is not None:
#         # 计算马氏距离
#         result = np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
#     else:
#         # 否则，计算欧氏距离
#         result = np.sqrt(np.dot(diff.T, diff))
#     # 返回距离值
#     return result

# def bbox2array(cls, bbox):
#     if bbox is None:
#         return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
#     else:
#         return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])
#
# def m_distance(det, trk, trk_inv_innovation_matrix=None):
#     """
#     Compute Mahalanobis distance between a detection and a track.
#
#     :type det: list[tlbr] | np.ndarray
#     :type trk: list[tlbr] | np.ndarray
#     :type trk_inv_innovation_matrix: np.ndarray | None
#
#     :rtype: np.ndarray
#     """
#     # Convert detection and track bounding boxes to arrays, taking the first 7 elements
#     det_array = bbox2array(det)[:7]
#     trk_array = bbox2array(trk)[:7]
#
#     # Compute the difference vector
#     diff = np.expand_dims(det_array - trk_array, axis=1)
#
#     # Correct the yaw angle difference
#     corrected_yaw_diff = diff_orientation_correction(diff[3])
#     diff[3] = corrected_yaw_diff
#
#     # Compute distance
#     if trk_inv_innovation_matrix is not None:
#         # Compute Mahalanobis distance
#         result = np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
#     else:
#         # Compute Euclidean distance
#         result = np.sqrt(np.dot(diff.T, diff))
#
#     # Return the distance as a numpy array
#     return np.array(result)


import numpy as np

def m_distances(atlbrs, btlbrs, trk_inv_innovation_matrix=None):
    """
    Compute cost based on Mahalanobis distance # 基于马氏距离计算距离矩阵

    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray
    :type trk_inv_innovation_matrix: np.ndarray | None

    :rtype: np.ndarray
    """
    dists = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float) # 创建一个形状为 (len(atlbrs), len(btlbrs))的全零矩阵，数据类型为 float
    if dists.size == 0:  # 如果矩阵为空，则将直接返回全零矩阵
        return dists

    # 内嵌函数：将边界框转换为数组
    def bbox2array(bbox):
        """
        将边界框转换为数组。
        :param bbox: 包含边界框信息的对象（长度为4的列表）
        :return: 包含边界框信息的 numpy 数组
        """
        return np.array(bbox)

    # 循环计算 atlbrs 和 btlbrs 之间的马氏距离
    for i, at in enumerate(atlbrs):
        for j, bt in enumerate(btlbrs):
            # 将检测框和跟踪框转换为数组
            det_array = bbox2array(at)
            trk_array = bbox2array(bt)

            # 计算检测框和跟踪框之间的差异向量
            diff = np.expand_dims(det_array - trk_array, axis=1)

            # 对偏航角差异进行修正
            corrected_yaw_diff = diff_orientation_correction(diff[3])
            diff[3] = corrected_yaw_diff

            # 计算距离
            if trk_inv_innovation_matrix is not None:
                # 计算马氏距离
                result = np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
            else:
                # 计算欧氏距离
                result = np.sqrt(np.dot(diff.T, diff))

            # 将结果存入距离矩阵
            dists[i, j] = result

    return dists

