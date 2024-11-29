import numpy as np

# 数据关联 方法  local iou  和  global iou  和原文 的 匈牙利算法（linear_assignment）相似
# # 使用 线性分配算法（匈牙利算法） 将轨迹与检测框 进行关联。    dists信息中 这个里面既包含轨迹也包含检测框信息 ？？
# matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

def GreedyAssignment(cost, threshold=None, method='global'):
    """
    使用IoU匹配进行线性分配

    :param cost: ndarray
        一个NxM的矩阵，表示每个track_ids与detection_ids之间的成本
    :param threshold: float
        如果成本大于阈值，则不予考虑
    :param method: str
        可选值: 'global', 'local'

    :rtype matches: ndarray
        匹配的轨迹和检测的索引对，形状为(N, 2)
    :rtype unmatched_a: tuple
        未匹配的轨迹的索引
    :rtype unmatched_b: tuple
        未匹配的检测的索引
    """

    cost_c = np.atleast_2d(cost)  # 将cost转换为至少二维的数组
    sz = cost_c.shape  # 获取cost_c的形状

    if threshold is None:
        threshold = 1.0

    matches = []  # 存储匹配的轨迹和检测的索引对
    unmatched_a = ()  # 存储未匹配的轨迹的索引
    unmatched_b = ()  # 存储未匹配的检测的索引

    if method == 'global':  # 全局方法
        vector_in = list(range(sz[0]))  # 创建包含轨迹索引的列表
        vector_out = list(range(sz[1]))  # 创建包含检测索引的列表
        while min(len(vector_in), len(vector_out)) > 0:  # 当轨迹和检测都还有剩余时
            v = cost_c[np.ix_(vector_in, vector_out)]  # 从cost_c中选择对应的轨迹和检测的成本
            min_cost = np.min(v)  # 找到成本的最小值

            if min_cost <= threshold:  # 如果最小成本小于等于阈值
                place = np.where(v == min_cost)  # 找到最小成本的位置
                matches.append([vector_in[place[0][0]], vector_out[place[1][0]]])  # 将匹配的轨迹和检测的索引对添加到matches中
                del vector_in[place[0][0]]  # 从vector_in中删除已匹配的轨迹索引
                del vector_out[place[1][0]]  # 从vector_out中删除已匹配的检测索引
            else:
                break  # 如果最小成本大于阈值，则终止循环

        unmatched_a_track = tuple(vector_in)  # 将剩余的轨迹索引转换为元组形式 track
        unmatched_b_detection = tuple(vector_out)  # 将剩余的检测索引转换为元组形式  detection
    else:  # 局部方法
        vector_in = []  # 存储未匹配的轨迹的索引
        vector_out = list(range(sz[1]))  # 创建包含检测索引的列表
        index = 0
        while min(sz[0] - len(vector_in), len(vector_out)) > 0:  # 当轨迹和检测都还有剩余时
            if index >= sz[0]:
                break
            place = np.argmin(cost_c[np.ix_([index], vector_out)])  # 找到成本最小的检测索引的位置

            if cost_c[index, vector_out[place]] <= threshold:  # 如果最小成本小于等于阈值
                matches.append([index, vector_out[place]])  # 将匹配的轨迹和检测的索引对添加到matches中
                del vector_out[place]  # 从vector_out中删除已匹配的检测索引
            else:
                vector_in.append(index)  # 将未匹配的轨迹索引添加到vector_in中
            index += 1  # 增加索引的值，以处理下一个轨迹

        vector_in += list(range(index, sz[0]))  # 将剩余的轨迹索引添加到vector_in中
        unmatched_a_track  = tuple(vector_in)  # 将剩余的轨迹索引转换为元组形式
        unmatched_b_detection = tuple(vector_out)  # 将剩余的检测索引转换为元组形式

    matches = np.asarray(matches)  # 将matches转换为NumPy数组

    return matches, unmatched_a_track, unmatched_b_detection  # 返回匹配的轨迹和检测的索引对，以及未匹配的轨迹和检测的索引