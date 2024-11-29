import numpy as np
import torch
import math
from cython_bbox import bbox_overlaps as bbox_ious


def Iou(box1, box2, wh=False):
    # 判断是否使用宽高表示的矩形框
    if wh == False:
        # 获取左上角和右下角的坐标
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        # 获取中心点坐标和宽高，转换为左上角和右下角的坐标
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算交集面积和交并比
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比

    return iou

#
# def Giou(rec1, rec2):
#     # 分别是第一个矩形左右上下的坐标
#     x1, x2, y1, y2 = rec1
#     x3, x4, y3, y4 = rec2
#
#     # 计算交并比
#     # iou = Iou(rec1, rec2) #
#     # iou = ious1(rec1, rec2) # 项目源码中的 iou
#     # 计算闭包区域的面积
#     area_C = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
#
#     # 计算两个矩形各自的面积
#     area_1 = (x2 - x1) * (y1 - y2)
#     area_2 = (x4 - x3) * (y3 - y4)
#
#     # 计算两个矩形的并集面积
#     sum_area = area_1 + area_2
#
#     # 计算第一个矩形的宽度和高度
#     w1 = x2 - x1
#     h1 = y1 - y2
#
#     # 计算第二个矩形的宽度和高度
#     w2 = x4 - x3
#     h2 = y3 - y4
#
#     # 计算交叉部分的宽度和高度
#     W = min(x1, x2, x3, x4) + w1 + w2 - max(x1, x2, x3, x4)
#     H = min(y1, y2, y3, y4) + h1 + h2 - max(y1, y2, y3, y4)
#
#     # 计算交叉部分的面积
#     Area = W * H
#
#     # 计算两个矩形并集的面积
#     add_area = sum_area - Area
#
#     # 计算闭包区域中不属于两个矩形的区域占闭包区域的比重
#     end_area = (area_C - add_area) / area_C
#
#     # 计算 GIoU（Generalized Intersection over Union）
#     giou = iou - end_area
#
#     return giou


def Giou(atlbrs, btlbrs):
    """
    基于GIoU（广义交并比）计算成本
    :param atlbrs: 边界框的tlbr列表
    :param btlbrs: 边界框的tlbr列表
    :return: giou np.ndarray
    """
    giou_scores = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)  # 创建一个形状为(len(atlbrs), len(btlbrs))的全零矩阵
    if giou_scores.size == 0:  # 如果矩阵为空，则返回全零矩阵
        return giou_scores

    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            giou_scores[i, j] = GiouSingle(atlbrs[i], btlbrs[j])  # 计算每对边界框的GIoU得分

    return giou_scores


def GiouSingle(rec1, rec2):
    """
    计算两个边界框之间的GIoU（广义交并比）
    :param rec1: 边界框的tlbr
    :param rec2: 边界框的tlbr
    :return: giou得分
    """
    x1, y1, x2, y2 = rec1  # 提取第一个边界框的坐标（左上角：(x1, y1)，右下角：(x2, y2)）
    x3, y3, x4, y4 = rec2  # 提取第二个边界框的坐标（左上角：(x3, y3)，右下角：(x4, y4)）

    # 计算交集面积
    intersection_w = max(0, min(x2, x4) - max(x1, x3))
    intersection_h = max(0, min(y1, y3) - max(y2, y4))
    intersection_area = intersection_w * intersection_h

    # 计算并集面积
    area_1 = (x2 - x1) * (y1 - y2)
    area_2 = (x4 - x3) * (y3 - y4)
    union_area = area_1 + area_2 - intersection_area

    # 计算包围面积
    enclosing_x1 = min(x1, x3)
    enclosing_y1 = max(y1, y3)
    enclosing_x2 = max(x2, x4)
    enclosing_y2 = min(y2, y4)
    enclosing_w = max(0, enclosing_x2 - enclosing_x1)
    enclosing_h = max(0, enclosing_y1 - enclosing_y2)
    enclosing_area = enclosing_w * enclosing_h

    # 计算IoU
    iou = intersection_area / union_area

    # 计算GIoU
    giou = iou - (enclosing_area - union_area) / enclosing_area

    return giou


#
# def Diou(bboxes1, bboxes2):
#     rows = bboxes1.shape[0]  # 获取矩形框1的行数（矩形框数量）
#     cols = bboxes2.shape[0]  # 获取矩形框2的行数（矩形框数量）
#     dious = torch.zeros((rows, cols))  # 创建一个全零张量用于存储计算得到的 Diou 值
#     if rows * cols == 0:  # 如果矩形框数量为0，则返回全零的 Diou 张量
#         return dious
#     exchange = False
#     if bboxes1.shape[0] > bboxes2.shape[0]:
#         bboxes1, bboxes2 = bboxes2, bboxes1  # 将矩形框数量较多的赋值给 bboxes2，确保 bboxes1 的行数不大于 bboxes2 的行数
#         dious = torch.zeros((cols, rows))  # 调整 Diou 张量的形状以匹配交叉计算的结果
#         exchange = True  # 记录是否进行了矩形框交换
#
#     # 计算矩形框的宽度和高度
#     w1 = bboxes1[:, 2] - bboxes1[:, 0]
#     h1 = bboxes1[:, 3] - bboxes1[:, 1]
#     w2 = bboxes2[:, 2] - bboxes2[:, 0]
#     h2 = bboxes2[:, 3] - bboxes2[:, 1]
#
#     # 计算矩形框的面积
#     area1 = w1 * h1
#     area2 = w2 * h2
#
#     # 计算矩形框的中心点坐标
#     center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
#     center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
#     center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
#     center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
#
#     # 计算矩形框的交集和外接框的坐标
#     inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
#     inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
#     out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
#     out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])
#
#     # 计算矩形框的交集面积和对角线距离
#     inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
#     inter_area = inter[:, 0] * inter[:, 1]
#     inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
#
#     # 计算矩形框的外接框和对角线距离
#     outer = torch.clamp((out_max_xy - out_min_xy), min=0)
#     outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
#
#     # 计算 Diou 值
#     union = area1 + area2 - inter_area
#     dious = inter_area / union - (inter_diag) / outer_diag
#     dious = torch.clamp(dious, min=-1.0, max=1.0)  # 将 Diou 值限制在 -1 到 1 之间
#
#     if exchange:
#         dious = dious.T  # 如果进行了矩形框交换，需要调整 Diou 张量的形状
#
#     return dious


def Diou(atlbrs, btlbrs):
    """
    根据 Diou 计算距离代价
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype dious np.ndarray
    """
    dious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)  # 创建一个形状为 (len(atlbrs), len(btlbrs)) 的全零矩阵，数据类型为 float
    if dious.size == 0:  # 如果矩阵为空，则直接返回全零矩阵
        return dious

    # 计算矩形框的宽度和高度
    w1 = atlbrs[:, 2] - atlbrs[:, 0]
    h1 = atlbrs[:, 3] - atlbrs[:, 1]
    w2 = btlbrs[:, 2] - btlbrs[:, 0]
    h2 = btlbrs[:, 3] - btlbrs[:, 1]

    # 计算矩形框的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算矩形框的中心点坐标
    center_x1 = (atlbrs[:, 2] + atlbrs[:, 0]) / 2
    center_y1 = (atlbrs[:, 3] + atlbrs[:, 1]) / 2
    center_x2 = (btlbrs[:, 2] + btlbrs[:, 0]) / 2
    center_y2 = (btlbrs[:, 3] + btlbrs[:, 1]) / 2

    # 计算矩形框的交集和外接框的坐标
    inter_max_xy = np.minimum(atlbrs[:, 2:], btlbrs[:, 2:])
    inter_min_xy = np.maximum(atlbrs[:, :2], btlbrs[:, :2])
    out_max_xy = np.maximum(atlbrs[:, 2:], btlbrs[:, 2:])
    out_min_xy = np.minimum(atlbrs[:, :2], btlbrs[:, :2])

    # 计算矩形框的交集面积和对角线距离
    inter = np.clip((inter_max_xy - inter_min_xy), a_min=0, a_max=None)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # 计算矩形框的外接框和对角线距离
    outer = np.clip((out_max_xy - out_min_xy), a_min=0, a_max=None)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)

    # 计算 Diou 值
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = np.clip(dious, a_min=-1.0, a_max=1.0)  # 将 Diou 值限制在 -1 到 1 之间

    return dious



#
# def bbox_overlaps_ciou(bboxes1, bboxes2):
#     rows = bboxes1.shape[0]  # 获取矩形框1的行数（矩形框数量）
#     cols = bboxes2.shape[0]  # 获取矩形框2的行数（矩形框数量）
#     cious = torch.zeros((rows, cols))  # 创建一个全零张量用于存储计算得到的 Ciou 值
#     if rows * cols == 0:  # 如果矩形框数量为0，则返回全零的 Ciou 张量
#         return cious
#     exchange = False
#     if bboxes1.shape[0] > bboxes2.shape[0]:
#         bboxes1, bboxes2 = bboxes2, bboxes1  # 将矩形框数量较多的赋值给 bboxes2，确保 bboxes1 的行数不大于 bboxes2 的行数
#         cious = torch.zeros((cols, rows))  # 调整 Ciou 张量的形状以匹配交叉计算的结果
#         exchange = True  # 记录是否进行了矩形框交换
#
#     # 计算矩形框的宽度和高度
#     w1 = bboxes1[:, 2] - bboxes1[:, 0]
#     h1 = bboxes1[:, 3] - bboxes1[:, 1]
#     w2 = bboxes2[:, 2] - bboxes2[:, 0]
#     h2 = bboxes2[:, 3] - bboxes2[:, 1]
#
#     # 计算矩形框的面积
#     area1 = w1 * h1
#     area2 = w2 * h2
#
#     # 计算矩形框的中心点坐标
#     center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
#     center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
#     center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
#     center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
#
#     # 计算矩形框的交集和外接框的坐标
#     inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
#     inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
#     out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
#     out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])
#
#     # 计算矩形框的交集面积和对角线距离
#     inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
#     inter_area = inter[:, 0] * inter[:, 1]
#     inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
#
#     # 计算矩形框的外接框和对角线距离
#     outer = torch.clamp((out_max_xy - out_min_xy), min=0)
#     outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
#
#     # 计算 IoU
#     union = area1 + area2 - inter_area
#     u = (inter_diag) / outer_diag
#     iou = inter_area / union
#
#     # 计算 CIOU
#     with torch.no_grad():
#         arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)  # 计算矩形框宽高比的差异
#         v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)  # 计算矩形框宽高比的差异的方差
#         S = 1 - iou  # 计算面积差异
#         alpha = v / (S + v)  # 计算调整系数 alpha
#         w_temp = 2 * w1  # 临时变量，用于计算 CIOU
#
#     # 计算 CIOU 中的 ar 部分
#     ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
#
#     # 计算 CIOU 值
#     cious = iou - (u + alpha * ar)
#     cious = torch.clamp(cious, min=-1.0, max=1.0)  # 将 CIOU 值限制在 [-1, 1] 范围内
#
#     if exchange:
#         cious = cious.T  # 如果进行了矩形框交换，将 CIOU 张量进行转置
#
#     return cious  # 返回计算得到的 CIOU 张量

# 此函数将计算基于Ciou的重叠度，输入为锚框和目标框的坐标，
# 形状分别为(N, 4)和(M, 4)，输出为一个形状为(N, M)的重叠度矩阵，其中每个元素表示对应锚框和目标框之间的Ciou值。

# def bbox_overlaps_ciou(atlbrs, btlbrs):
#     """
#     计算基于Ciou的重叠度
#
#     :param atlbrs: 锚框（anchor）的左上角和右下角坐标，形状为 (N, 4)，N为锚框数量
#     :type atlbrs: torch.Tensor
#
#     :param btlbrs: 目标框（target）的左上角和右下角坐标，形状为 (M, 4)，M为目标框数量
#     :type btlbrs: torch.Tensor
#
#     :return: 重叠度矩阵，形状为 (N, M)
#     :rtype: torch.Tensor
#     """
#     rows = atlbrs.shape[0]  # 获取锚框的数量
#     cols = btlbrs.shape[0]  # 获取目标框的数量
#     ciou = torch.zeros((rows, cols))  # 创建一个全零张量用于存储计算得到的Ciou值
#
#     if rows * cols == 0:  # 如果锚框数量或目标框数量为0，则返回全零的Ciou张量
#         return ciou
#
#     exchange = False
#     if rows > cols:
#         atlbrs, btlbrs = btlbrs, atlbrs  # 将目标框数量较多的赋值给btlbrs，确保atlbrs的行数不大于btlbrs的行数
#         ciou = torch.zeros((cols, rows))  # 调整Ciou张量的形状以匹配交叉计算的结果
#         exchange = True  # 记录是否进行了框的交换
#
#     # 计算框的宽度和高度
#     aw = atlbrs[:, 2] - atlbrs[:, 0]
#     ah = atlbrs[:, 3] - atlbrs[:, 1]
#     bw = btlbrs[:, 2] - btlbrs[:, 0]
#     bh = btlbrs[:, 3] - btlbrs[:, 1]
#
#     # 计算框的面积
#     area_a = aw * ah
#     area_b = bw * bh
#
#     # 计算框的中心点坐标
#     center_ax = (atlbrs[:, 2] + atlbrs[:, 0]) / 2
#     center_ay = (atlbrs[:, 3] + atlbrs[:, 1]) / 2
#     center_bx = (btlbrs[:, 2] + btlbrs[:, 0]) / 2
#     center_by = (btlbrs[:, 3] + btlbrs[:, 1]) / 2
#
#     # 计算框的交集和外接框的坐标
#     inter_max_xy = torch.min(atlbrs[:, 2:], btlbrs[:, 2:])
#     inter_min_xy = torch.max(atlbrs[:, :2], btlbrs[:, :2])
#     outer_max_xy = torch.max(atlbrs[:, 2:], btlbrs[:, 2:])
#     outer_min_xy = torch.min(atlbrs[:, :2], btlbrs[:, :2])
#
#     # 计算交集的宽度和高度
#     inter_wh = torch.clamp((inter_max_xy - inter_min_xy), min=0)
#
#     # 计算交集的面积
#     inter_area = inter_wh[:, 0] * inter_wh[:, 1]
#
#     # 计算外接框的宽度和高度
#     outer_wh = torch.clamp((outer_max_xy - outer_min_xy), min=0)
#
#     # 计算外接框的对角线距离的平方
#     outer_diag_sq = (outer_wh[:, 0] ** 2) + (outer_wh[:, 1] ** 2)
#
#     # 计算IoU
#     union = area_a + area_b - inter_area
#     iou = inter_area / union
#
#     # 计算CIOU
#     with torch.no_grad():
#         arctan = torch.atan(bw / bh) - torch.atan
#         arctan_ = torch.atan((aw + 1e-6) / (ah + 1e-6))
#         v = (4 / (math.pi ** 2)) * torch.pow((arctan - arctan_), 2)  # 计算v值
#
#         S = 1 - iou  # 计算S值
#
#         alpha = v / (S + v)  # 计算alpha值
#
#         ciou = iou - (alpha * v)  # 计算Ciou值
#
#     if exchange:
#         ciou = ciou.T  # 如果进行了框的交换，则将Ciou张量转置回原始形状
#
#     return ciou

import torch
import math
import numpy as np

def bbox_overlaps_ciou(bboxes1, bboxes2):
    """
    计算两组边界框之间的 Complete IoU (CIoU)。

    Args:
    - bboxes1 (np.ndarray): 第一组边界框，每行包含四个值 [top, left, bottom, right]。
    - bboxes2 (np.ndarray): 第二组边界框，每行包含四个值 [top, left, bottom, right]。

    Returns:
    - cious (np.ndarray): CIoU 矩阵，大小为 (len(bboxes1), len(bboxes2))。
    """

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = np.zeros((rows, cols), dtype=np.float)

    if rows * cols == 0:
        return cious

    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = np.zeros((cols, rows), dtype=np.float)
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = np.maximum(bboxes1[:, :2], bboxes2[:, :2])
    inter = np.clip((inter_max_xy - inter_min_xy), a_min=0, a_max=None)
    inter_area = inter[:, 0] * inter[:, 1]

    union = area1 + area2 - inter_area
    iou = inter_area / union

    with np.errstate(divide='ignore', invalid='ignore'):
        arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
        v = (4 / (np.pi ** 2)) * (arctan ** 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
        ar = (8 / (np.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
        cious = iou - (u + alpha * ar)
        cious = np.clip(cious, -1.0, 1.0)

    if exchange:
        cious = cious.T

    return cious

# 添加中文注释
"""
上述函数计算两组边界框之间的 Complete IoU (CIoU)。

计算过程包括以下步骤：
1. 计算两组边界框的宽度和高度。
2. 计算两组边界框的面积。
3. 计算两组边界框的中心点坐标。
4. 计算交集区域的最大和最小坐标，并计算交集面积。
5. 计算并修正并集面积和 IoU。
6. 计算辅助参数 alpha 和 ar。
7. 计算最终的 CIoU 值，并对其进行限制在 [-1.0, 1.0] 范围内。
"""

