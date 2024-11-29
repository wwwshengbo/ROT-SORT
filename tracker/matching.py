import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter

# from other_IOU import Giou, Diou, bbox_overlaps_ciou
from .other_IOU import Giou,bbox_overlaps_ciou,Diou
from .other_utils1 import  m_distances


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    # lap.lapjv 是一个来自 lap 库的函数，通常用于求解线性分配问题（既匈牙利算法的一种实现）
    # cost_matrix 表示任务分配的成本矩阵, extend_cost=True 一个布尔值，表示是否扩展本矩阵以确保矩阵是方形的, cost_limit=thresh  一个阈值表示接受匹配的最大成本
    # 如果没有传递 thresh 参数，那么它将默认为 float('inf') （无穷大），这意味着如果没有显示传递阈值，则所有可能都会被接受，因为没有任何成本会大于无穷大
    # cost 总匹配成本, x 每个任务对应的分配, y 每个分配对应的任务
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # 枚举 x 中的每个元素及其索引
    for ix, mx in enumerate(x):
        # 检查mx 是否大于0 ，表示找到匹配
        if mx >= 0:
            # 将匹配对 （ix，mx）添加到 matches 列表中
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]  # 找到x 中小于0的索引，这些索引表示未匹配的元素
    unmatched_b = np.where(y < 0)[0]  # 找到y 中小于0的索引，这些索引表示未匹配的元素
    matches = np.asarray(matches)  # 将 matches 列表转换为 numpy 数组
    return matches, unmatched_a, unmatched_b  # 返回 匹配对，未匹配的元素a 和 b的 索引


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU # 基于iou计算 距离矩阵
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float) # 创建一个形状为 (len(atlbrs), len(btlbrs))的全零矩阵，数据类型为 float
    if ious.size == 0:  # 如果矩阵为空，则将直接返回 全零矩阵
        return ious
    # 调用 bbox_ious 函数 计算 atlbrs 和 btlbrs 之间的iou
    ious = bbox_ious(
        # np.ascontiguousarray 函数的作用是创建一个连续的内存布局的数组，并返回该数组。arr：要转换为连续内存布局的数组。
        # dtype（可选）：指定返回数组的数据类型。如果未提供该参数，则保持输入数组的数据类型。
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )
    # print("SHAPE",ious.shape)  # (5, 5) 、(1, 1)、(6, 7) 等
    return ious    # 返回得到的iou矩阵


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU  基于IOU 计算距离矩阵
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    # 检查 atracks 与 btracks 是否包含 np.ndarray 元素
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        # 如果是，将atracks 和 btracks 分别赋值给  atlbrs 和  btlbrs
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 如果不是，从 atracks 和 btlbrs 中提取每个轨迹的 tlbr(边界框)
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # 计算 atlbrs 和 btlbrs 两个边界框之间的 iou， key1
    # print('atlbrs', atlbrs)
    # print('btlbrs', btlbrs)  # 输出多个一维数组，每个数组包含 4个数值，代表一个边界框信息
    # print('btlbrs', btlbrs)
    _ious = ious(atlbrs, btlbrs)
    # print('_ious', _ious)
    # 1减去 iou值的目的是将 iou值转换为距离度量，这是因为距离度量通常表示两个对象之间的差异程度，而iou值表示两个边界框之间的重叠程度
    cost_matrix = 1 - _ious # 将距离定义为1 减去 iou，计算距离矩阵

    return cost_matrix  #返回计算得到的距离矩阵 作为 iou距离的结果



# iou 距离 使用 giou 计算
def iou_distance1(atracks, btracks):
    """
    Compute cost based on IoU  基于IOU 计算距离矩阵
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    # 检查 atracks 与 btracks 是否包含 np.ndarray 元素
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        # 如果是，将atracks 和 btracks 分别赋值给  atlbrs 和  btlbrs
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 如果不是，从 atracks 和 btlbrs 中提取每个轨迹的 tlbr(边界框)
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # 计算 atlbrs 和 btlbrs 两个边界框之间的 iou， key1
    # print('atlbrs', atlbrs)
    # print('btlbrs', btlbrs)  # 输出多个一维数组，每个数组包含 4个数值，代表一个边界框信息
    # print('btlbrs', btlbrs)
    _ious = Giou(atlbrs, btlbrs)
    # print('_ious', _ious)
    # 1减去 iou值的目的是将 iou值转换为距离度量，这是因为距离度量通常表示两个对象之间的差异程度，而iou值表示两个边界框之间的重叠程度
    cost_matrix = 1 - _ious # 将距离定义为1 减去 iou，计算距离矩阵

    return cost_matrix  #返回计算得到的距离矩阵 作为 iou距离的结果


def compute_m_distance(dets, tracks, trk_innovation_matrix):
    """ compute l2 or mahalanobis distance
        when the input trk_innovation_matrix is None, compute L2 distance (euler)
        else compute mahalanobis distance
        return dist_matrix: numpy array [len(dets), len(tracks)]
    """
    euler_dis = (trk_innovation_matrix is None)  # 是否使用欧氏距离

    # 如果不是欧氏距离，则计算马氏距离需要先计算每个跟踪目标的逆创新矩阵
    if not euler_dis:
        trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]

    # 初始化距离矩阵，大小为 [len(dets), len(tracks)]
    dist_matrix = np.empty((len(dets), len(tracks)),dtype=np.float)

    # 遍历每个检测到的物体（dets）
    for i, det in enumerate(dets):
        # 遍历每个跟踪目标（tracks）
        for j, trk in enumerate(tracks):
            if euler_dis:
                # 如果使用欧氏距离，直接调用工具函数 utils.m_distance 计算欧氏距离
                dist_matrix[i, j] = m_distances(det, trk)
            else:
                # 如果使用马氏距离，调用工具函数 utils.m_distance 计算马氏距离，
                # 需要传入对应的逆创新矩阵 trk_inv_inn_matrices[j]
                dist_matrix[i, j] = m_distances(det, trk, trk_inv_inn_matrices[j])

    # 返回计算得到的距离矩阵
    return dist_matrix



def iou_distance2(atracks, btracks):
    """
    Compute cost based on IoU  基于IOU 计算距离矩阵
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    # 检查 atracks 与 btracks 是否包含 np.ndarray 元素
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        # 如果是，将atracks 和 btracks 分别赋值给  atlbrs 和  btlbrs
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 如果不是，从 atracks 和 btlbrs 中提取每个轨迹的 tlbr(边界框)
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # 计算 atlbrs 和 btlbrs 两个边界框之间的 iou， key1
    # print('atlbrs', atlbrs)
    # print('btlbrs', btlbrs)  # 输出多个一维数组，每个数组包含 4个数值，代表一个边界框信息
    # print('btlbrs', btlbrs)
    _ious = bbox_overlaps_ciou(atlbrs, btlbrs)
    # print('_ious', _ious)
    # _ious = Diou(atlbrs, btlbrs)
    # 1减去 iou值的目的是将 iou值转换为距离度量，这是因为距离度量通常表示两个对象之间的差异程度，而iou值表示两个边界框之间的重叠程度
    cost_matrix = 1 - _ious # 将距离定义为1 减去 iou，计算距离矩阵

    return cost_matrix  #返回计算得到的距离矩阵 作为 iou距离的结果


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU  基于iou 计算距离矩阵
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    # 检查 atracks 和 btracks 是否包含 np.ndarray 元素
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        # 如果是，将 atracks 和 btracks 分别赋值给 atlbrs 和 btlbrs
        atlbrs = atracks
        btlbrs = btracks
    else:
        #如果不是，从atracks 和 btracks 中提取每个轨迹 tlwh_to_tlbr(track.pred_bbox) 作为边界框信息
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs) # 计算 atlbrs 和 btlbrs 之间的iou距离
    cost_matrix = 1 - _ious  # 将距离定义为1 减去iou，计算距离矩阵

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)


    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


# # 对上一行得到的距离矩阵，进行门控处理，使用卡尔玛滤波进一步修正距离矩阵，排除不太可能得匹配
def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


# 下面是第三次 匹配单独使用到 包含运动阈值？？
# 第三次 匹配时的 使用到的 ，计算高斯距离
def motion_distance(kf,tracks, detections,only_position=False, motion_thresh=100):

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if len(detections) != 0 :
        gating_threshold = motion_thresh*motion_thresh
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='gaussian')
            cost_matrix[row] = gating_distance/gating_threshold
            cost_matrix[row, gating_distance > gating_threshold] = 1
    return cost_matrix

# 第三次 匹配时使用到，根据类别、宽度和高度 融合距离
def fuse_classes_width_height(cost_matrix, strack_pool,detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix

    strack_classes = np.array([strack.classes for strack in strack_pool])
    det_classes = np.array([det.classes for det in detections])

    strack_tlwh = np.array([strack.tlwh for strack in strack_pool])
    det_tlwh = np.array([det.tlwh for det in detections])

    classes_width_height_matrix = iou_sim.copy()
    for i in range(strack_classes.size):
        for j in range(det_classes.size):
            if strack_classes[i] == det_classes[j] and (1/4 < (strack_tlwh[i,2]*strack_tlwh[i,3])/(det_tlwh[j,2]*det_tlwh[j,3]) < 4):
                classes_width_height_matrix[i,j] = 1
            else:
                classes_width_height_matrix[i,j]=0

    fuse_sim = iou_sim * classes_width_height_matrix
    fuse_cost = 1 - fuse_sim
    return fuse_cost

# 按照上面修改一下，把类别去掉，gtp 修改的代码
# 第三次 匹配时使用到，根据宽度和高度 融合距离
def fuse_width_height(cost_matrix, strack_pool, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix

    strack_tlwh = np.array([strack.tlwh for strack in strack_pool])
    det_tlwh = np.array([det.tlwh for det in detections])

    width_height_matrix = iou_sim.copy()
    for i in range(strack_tlwh.shape[0]):
        for j in range(det_tlwh.shape[0]):
            if 1/4 < (strack_tlwh[i,2]*strack_tlwh[i,3])/(det_tlwh[j,2]*det_tlwh[j,3]) < 4:
                width_height_matrix[i,j] = 1
            else:
                width_height_matrix[i,j] = 0

    fuse_sim = iou_sim * width_height_matrix
    fuse_cost = 1 - fuse_sim
    return fuse_cost

# 第三次匹配使用，使用线性分配算法 匹配轨迹和检测框 ，与前面相比多阈值
def linear_assignment_thresh(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


