import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching

from tracker import other_matching_1

from tracker.gmc import GMC    
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter  

from fast_reid.fast_reid_interfece import FastReIDInterface  
from yolox.utils.visualize import plot_tracking,plot_tracking_custom,plot_tracking_customRG,plot_tracking_custom1,plot_tracking_customT


# 定义一个STrack 的类，它继承自BaseTrack 类，既STrack 是BaseTrack 的子类
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()   # shared_kalman 是一个类属性，表示一个共享的卡尔曼滤波器(KalmanFilter)，  它可能是一个全局的卡尔曼滤波器实例，用于在多个STrack 对象之间共享

    def __init__(self, tlwh, score, feat=None, feat_history=50):
        # 定义STrack 类的初始化方法（构造函数）接收以下参数：
        # tlwh：一个列表，表示轨迹的位置和尺寸信息 （top_left_x,top_left_y,width,height）
        # score : 一个浮点数，轨迹的分数。feat：一个特征向量，表示轨迹的特征信息，默认为 none。feat_history: 一个整数，表示保留的特征向量的历史长度，默认为50
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)  # 传入一个tlwh 参数转换为一个 numpy 数组，并保存在_tlwh 属性中。 _tlwh 是轨迹的位置和尺寸信息的内部变量
        # 下面两行，这些变量用于存储 卡尔曼滤波器的状态信息，在初始化时，它们被设置为 None
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False   # is_activated 是一个布尔值，表示跟踪目标是否已激活，在初始化时，设置为 False

        self.score = score # 将传入的 score 参数 赋值给 score 属性，表示轨迹分数
        self.tracklet_len = 0 # tracklet_len 是一个整数，表示轨迹长度（目标已被跟踪的帧数），初始化设置为 0

        # 经过平滑处理的 embeding feature # smooth_feat 和 用于存储，平滑处理的特征向量 和当前特征向量，初始化时为 None
        self.smooth_feat = None
        self.curr_feat = None
        self.track_high_thresh =0.6

        # update embeding feature 更新嵌入特征
        if feat is not None:
            self.update_features(feat)  # 如果 feat 不为 None 调用update_features 方法更新嵌入特征
        self.features = deque([], maxlen=feat_history)  # 创建一个双端队列（deque）对象，并将其赋值给self.features 属性。
        # 双端队列用于存储特征向量的历史记录，并且 maxlen 参数指定了队列的最大长度，既保留特征历史记录的数量
        # 平滑系数
        self.alpha = 0.9   # 原来代码，这个效果比0.95好点
        # self.alpha = 0.95  # 模仿 deep-oc-sort 参数，如果不行可以适当调小？？
        # self.track_high_thresh = 0.6




    # Strack 中 也增加了处理 embeding feature 方法
    # update_features 函数使用了3次，STrack 类中初始化 41 行、轨迹重新激活134行 、STrack 类中的 update 更新方法 160行
    def update_features(self, feat, alpha=0.9):
        feat /= np.linalg.norm(feat)  # 将特征向量 feat 进行归一化处理，除以特征向量的范数（欧几里得范数）
        self.curr_feat = feat # 将归一化的特征向量赋值给 self.curr_feat，表示当前的特征向量

        if self.smooth_feat is None:   # 检查 之前的 self.smooth_feat  平滑特征 是否为None ，如果是None
            self.smooth_feat = feat  # 将当前特征向量 feat 赋值给self.smooth_feat
        else:
            #如果不为 None 用 指数平滑 对 tracklet 的 feature 进行处理
            # 将当前的特征向量 feat 与之前的 平滑特征向量 self.smooth_feat 按照权重 self.alpha 进行线性组合，得到新的平滑特征向量
            # 上面解释不全，下面为 EMA(Exponential Moving Average 指数平均移动)计算公式，self.alpha 为平滑系数， self.smooth_feat 为当前re-id 分数 ， feat 为前一时刻的EMA值
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat # 将当前的特征向量 feat 与之前的 平滑特征向量 self.smooth_feat
        self.features.append(feat)   # 将归一化的特征向量 feat 添加到特征历史记录中，既 存储在 self.features 的双端队里中
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  # 接着对 平滑特征向量self.smooth_feat 惊醒归一化处理，除以其范数，以确保特征向量的单位长度

    def predict(self):
        mean_state = self.mean.copy()   # 复制均值状态向量
        if self.state != TrackState.Tracked:  #如果轨迹状态不是 Tracked ，将速度设为 0
            mean_state[6] = 0
            mean_state[7] = 0
        # 使用卡尔曼滤波器进行预测，并返回更新后的， 均值状态向量和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:           # 如果存在 stracks  （传入的 stracks 是 轨迹池 包含已跟踪 和 丢失的轨迹）
            multi_mean = np.asarray([st.mean.copy() for st in stracks]) # 复制所有的 stracks 的状态均值和 协方差 矩阵
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):                            # 遍历每个 strack （轨迹）
                if st.state != TrackState.Tracked:                      # 如果 strack 的状态不是 Tracked，将速度设置为 0
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            # 这个均值与协方差可以改？？扰动
            # multi_predict 是一个静态类或方法，对多个目标的状态进行同时预测
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)   # 使用共享的卡尔曼滤波器 进行多目标跟踪
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):                              # 更新每个strack 的均值状态向量 和协方差矩阵
                stracks[i].mean = mean                                                                       # 将状态向量和 协方差矩阵，分别更新会原始的 stracks 中的每个STrack对象
                stracks[i].covariance = cov

    # 增加了 global motion compensation (全局运动补偿) 的方法：multi_gmc
    # 根据 估计的全局运动 （由矩阵 H 表示）对目标轨迹进行校正，以减少运动照成的不确定性。
    # 通过将运动应用到 目标轨迹的均值状态向量 和协方差矩阵上， 可以更准确的预测 和 估计目标的位置 和 运动状态。
    # 原理是 什么？？？
    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:  # 如果轨迹池中 存在 轨迹
            # 未被调整过的kalman filter 参数
            multi_mean = np.asarray([st.mean.copy() for st in stracks])   # 复制每个 strack(轨迹)的均值状态向量
            multi_covariance = np.asarray([st.covariance for st in stracks]) # 复制每个 strack(轨迹)的协方差矩阵

            # 提取出 M矩阵
            R = H[:2, :2]   # 提取出 H 的前两行两列部分作为 R矩阵
            # 构造 M tiuta
            R8x8 = np.kron(np.eye(4, dtype=float), R)   # 构造 8x8 的R矩阵
            # 提取出T矩阵
            t = H[:2, 2] # 提取出 H 的前两行第三列部分 作为 t 向量
            # 调整 kalman filter 参数
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)    # 通过 R8x8 矩阵 对mean 进行调整
                mean[:2] += t           # 调整 mean 的前两个元素位置 加上 t 向量
                cov = R8x8.dot(cov).dot(R8x8.transpose())   # 通过 R8x8矩阵对 cov 进行调整

                stracks[i].mean = mean   # 更新strack 的均值状态向量
                stracks[i].covariance = cov   # 更新stack 的协方差 矩阵

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter   # 设置跟踪器的 卡尔曼滤波器
        self.track_id = self.next_id()       # 分配一个新的唯一跟踪id  # 这行代码实现 id 不断增加

        # 使用 卡尔曼滤波器 对初始位置和 尺寸进行初始化，得到均值状态向量和协方差矩阵
        # 使用卡尔曼滤波器的 initiate 方法，将 以 tlwh 格式 表示的初始位置和 尺寸转换为 以 xywh 格式表示，然后进行初始化，得到跟踪器的均值状态向量mean 和协方差矩阵covariance
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0       # 跟踪器的轨迹长度  初始化为 0
        self.state = TrackState.Tracked  # 设置跟踪器的状态为 已跟踪
        if frame_id == 1:
            self.is_activated = True   # 如果是第一帧 ，则将 跟踪器的激活状态设置为 True
        self.frame_id = frame_id   # 设置跟踪器的帧 id
        self.start_frame = frame_id   # 设置跟踪器的 起始帧 id

    # re_activate 方法接受三个参数，new_track,frame_id, new_id,其中 new_track 是新的跟踪器对象,frame_id 当前帧id, new_id 一个布尔值，表示是否分配新的跟踪id
    def re_activate(self, new_track, frame_id, new_id=False):
        # 使用卡尔曼滤波器的update 方法 根据  new_track.tlwh信息进行更新 均值状态向量 和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        # 添加当前检测的置信度分数， 自适应调节 卡尔曼滤波器每个状态变量对应的标准差
        # self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh), new_track.score) # 添加后分数下降
        if new_track.curr_feat is not None:     # 如果 new_track 的 curr_feat （特征向量） 不为空，则更新跟踪器的特征向量

            # 依据当前目标置信度 分数，自适应调整平滑系数
            trust = (new_track.score - self.track_high_thresh) / (1- self.track_high_thresh)
            af = self.alpha
            dets_alpha = af + (1-af) *(1-trust)
            # print("alpha 4:",  "new_track.score 4",new_track.score, "dets_alpha4", dets_alpha)
            self.update_features(new_track.curr_feat)  # 原论文代码
            # self.update_features(new_track.curr_feat,dets_alpha)  # 修改后的代码


        self.tracklet_len = 0       # 跟踪器的轨迹长度重新设置为 0
        self.state = TrackState.Tracked  # 设置跟踪器的 状态 为已跟踪
        self.is_activated = True  # 设置跟踪器的激活状态为 True
        self.frame_id = frame_id   # 设置跟踪器的 帧 id
        if new_id:
            self.track_id = self.next_id()  # 如果 new_id 为 True ，则分配一个新的 唯一跟踪id
        self.score = new_track.score  # 使用新的 new_track 的得分 来跟新跟踪器的得分

    def update(self, new_track, frame_id):
        """
        update 方法接收两个参数 new_track 和 frame_id 表示新的跟踪器对象 和当前帧id
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id   # 设置跟踪帧的 id 为当前id
        self.tracklet_len += 1 # 跟踪器的轨迹长度加1

        new_tlwh = new_track.tlwh  # 获取新跟踪器的tlwh (Top left ,width,height)信息
        # 使用 # 使用卡尔曼滤波器的update 方法 根据  new_tlwh信息进行更新 均值状态向量 和协方差矩阵.这里 通过 tlwh_to_xywh 方法将tlwh 格式 转化为 xywh格式
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))
        #  # 添加 当前目标置信度分数，以便修 自适应 卡尔曼滤波器每个状态变量对应的标准差 矩阵
        # self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh),new_track.score)

        if new_track.curr_feat is not None:  # 如果跟踪器的 curr_feat 不为空，则更新跟踪器的特征向量
            # 依据当前目标置信度 分数，自适应调整平滑系数
            trust = (new_track.score - self.track_high_thresh) / (1- self.track_high_thresh)
            # trust = (new_track.score - 0.6) / (1 - self.track_high_thresh)
            af = self.alpha
            dets_alpha = af + (1-af) *(1-trust)
            # print("alpha 5:","new_track.score 5",new_track.score, "dets_alpha5", dets_alpha)
            self.update_features(new_track.curr_feat) # 原论文代码
            # self.update_features(new_track.curr_feat,dets_alpha)  # 修改后的代码

        self.state = TrackState.Tracked   # 设置更新器的状态为 “已跟踪”
        self.is_activated = True   # 设置跟踪器的激活状态为 True

        self.score = new_track.score  # 使用新跟踪器的得分 来更新跟踪器的得分


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()   # 如果均值状态向量 mean 为空（既跟踪器尚未初始化），则返回原始的tlwh信息的 副本
        ret = self.mean[:4].copy()     # 赋值均值状态向量的前四个元素 到ret中，表示位置信息 （top left x， top left y， width，height）
        ret[:2] -= ret[2:] / 2         # 根据位置信息 调整 top left x 和 top left y 的值，使其表示左上角坐标
        return ret                     # 返回调整后的位置信息

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        将跟踪器的位置细腻转换为 (min x, min y, max x, max y) 的格式 表示跟踪器的 左上角和右下角，即 (top left, bottom right)。具体解释如下
        """
        ret = self.tlwh.copy()  # 复制跟踪器的位置信息 tlwh
        ret[2:] += ret[:2]      # 将位置信息中 width 和 height 分别加上 top left x 和 top left y, width, height 得到 右下角坐标
        return ret   # 返回转换后的位置信息

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        用于将跟踪器的位置信息转换为 (min x, min y, max x, max y) 的格式，即 (top left, bottom right)
        """
        ret = self.tlwh.copy()   # 复制跟踪器的位置信息 tlwh
        ret[:2] += ret[2:] / 2.0 # 将位置信息中的 前两个元素 ，top left x 和 top left y 分别加上  后两个两个元素 width 和 height 的一半，得到中心点的坐标
        return ret   # 返回转换后的位置信息，左上角 和 右下角 坐标

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        用于将边界框的位置信息从 (top left x, top left y, width, height) 转换为 (center x, center y, aspect ratio, height) 的格式
        """
        ret = np.asarray(tlwh).copy()  # 将 tlwh 转换为 numpy  数组 并进行复制
        ret[:2] += ret[2:] / 2         # 将位置信息中的 前两个元素 top left x 和 top left y 分别加上 后两个元素 width 和 height 的一半，得到中心点的坐标
        ret[2] /= ret[3]               # 将位置信息中的 width 除以 height ，得到宽高比
        return ret                     # 返回转换后的位置信息 表示边界框的 (center x, center y, aspect ratio, height)


    # 添加坐标转换，为第三次匹配使用
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()   # 将tlwh 转换为 numpy 数组并进行赋值
        ret[:2] += ret[2:] / 2   # 将位置信息中的 前两个元素 top left x 和 top left y 分别 加上 width 和 height 的一半，得到中心点的坐标
        return ret  # 返回转换后的位置信息  示边界框的 (center x, center y, width, height)

    def to_xywh(self):
        # 用于将跟踪器的位置信息从 (top left x, top left y, width, height)
        # 转换为 (center x, center y, width, height) 的格式。它使用了之前定义的 tlwh_to_xywh 方法来完成转换。
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        # 静态方法 tlbr_to_tlwh 接受一个参数 tlbr，表示边界框的位置信息
        # 用于将边界框的位置信息从 (top left x, top left y, bottom right x, bottom right y) 转换为 (top left x, top left y, width, height) 的格式
        ret = np.asarray(tlbr).copy()   # 将参数 tlbr 转换为 numpy 数组，并将其复制给变量ret
        # 计算边界框的宽度和高度，通过右下角坐标，减去左上角坐标来实现 。
        # 将位置信息中后两个元素  bottom right x 和 bottom right y 减去 前两个元素 top left x 和 top left y 得到边界框的宽度和高度
        ret[2:] -= ret[:2]
        return ret   # 返回转换后的位置信息 ret，表示边界框的 (top left x, top left y, width, height)

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        # 用于将边界框的位置信息从 (top left x, top left y, width, height) 转换为
        # (top left x, top left y, bottom right x, bottom right y) 的格式
        # 将tlwh 转换为numpy 数组，并创建副本
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2] # 将后两个元素 width 和 height ，加上前两个元素  top left x 和 top left y 得到 边界框右下角坐标
        return ret  # 转换后的结果 表示边界框的 (top left x, top left y, bottom right x, bottom right y)

    def __repr__(self):
        # 一个特殊方法  __repr__(self)，用于返回跟踪器 的字符串表示形式。它返回一个格式 化 的字符串，包含跟踪器 id ，起始帧 和结束 帧 信息
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

    # 第一次关联使用的阈值扰动范围
    def adaptive_threshold(score, min_score=0.90, max_score=0.96, min_thresh=0.7, max_thresh=0.99):
        # 将分数限制在 min_score 和 max_score 范围内
        score = max(min(score, max_score), min_score)

        # 使用映射计算阈值
        ratio = (score - min_score) / (max_score - min_score)
        threshold = min_thresh + ratio * (max_thresh - min_thresh)
        return threshold


    def adaptive_threshold_1(score, min_score=0.8, max_score=0.98, min_thresh=0.5, max_thresh=0.6):
        # 将分数限制在 min_score 和 max_score 范围内
        score = max(min(score, max_score), min_score)

        # 使用映射计算阈值
        ratio = (score - min_score) / (max_score - min_score)

        # 在 ratio 基础上添加扰动使阈值更好地分布在 min_thresh 和 max_thresh 之间
        perturbation = (1 - ratio) * 0.1  # 添加扰动以改变线性映射结果
        # threshold = min_thresh + (ratio + perturbation) * (max_thresh - min_thresh)
        threshold = min_thresh +  perturbation * (max_thresh - min_thresh)
        # print("001 threshold", threshold)
        # threshold = min(max(threshold, min_thresh), max_thresh)  # 确保阈值在 min_thresh 和 max_thresh 之间

        return threshold

    # 第三次关联使用阈值扰动范围
    def adaptive_threshold_2(score, min_score=0.60, max_score=0.98, min_thresh=0.5, max_thresh=0.98):
        # 将分数限制在 min_score 和 max_score 范围内
        score = max(min(score, max_score), min_score)

        # 使用映射计算阈值
        ratio = (score - min_score) / (max_score - min_score)
        threshold = min_thresh + ratio * (max_thresh - min_thresh)
        return threshold



# 在主类中，先行定义，基本上新添加的属性，为处理embeding feature (将原始数据转化为具有固定长度的向量表示过程) 和 global motion compensation 有关
# embeding feature (将原始数据转化为具有固定长度的向量表示过程，该向量被称为，嵌入向量 或特征向量，嵌入向量的维度通常是预先定义好的，并且能够捕捉到原始数据的重要特征信息)

class ROTSORT(object):
    # 初始化函数。它接受两个参数： args 是一个包含算法参数对象， frame_rate 是视频的帧率 ，默认30帧/秒
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]  # 初始化一个空列表，存储当前跟踪的目标，已跟踪轨迹（STrack）
        self.lost_stracks = []  # type: list[STrack]  # 。。，存储当前跟踪目标的丢失轨迹(STrack)，既跟踪目标在一定时间内没有被成功检测到
        self.removed_stracks = []  # type: list[STrack] # 。。用于存储被移除的轨迹（STrack），既跟踪目标被判定为无效或超出跟踪范围被移除
        BaseTrack.clear_count() # 调用BaseTrack 类的静态方法 clear_count() 用于重置目标计数器

        self.frame_id = 0   # 初始化当前帧的帧id，初始值为0
        self.args = args  # 将传入的args 参数赋值给实例变量self.args，用于在类的其它地方访问算法参数

        self.track_high_thresh = args.track_high_thresh  # 将args 中 的track_high_thresh 参数赋值给实例变量track_high_thresh ，用于指定 高阈值
        self.track_low_thresh = args.track_low_thresh  # 。。用于指定低阈值
        self.new_track_thresh = args.new_track_thresh  # 。。用于指定新目标阈值

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)  # 计算并初始化一个缓冲区大小，用于存储历史帧的轨迹信息。该缓冲区大小基于帧率和track_buffer 参数
        self.max_time_lost = self.buffer_size    # 将缓冲区大小赋值给实例变量，用于指定跟踪目标在没有被检测到的最大时间
        self.kalman_filter = KalmanFilter()   # 初始化一个KalmanFilter 对象，用于进行卡尔曼滤波预测和更新

        # ReID module
        self.proximity_thresh = args.proximity_thresh  # 将args 中的 proximity_thresh  参数赋值给实例变量 self.proximity_thresh，用于指定邻阈值
        self.appearance_thresh = args.appearance_thresh  # 将args 中 appearance_thresh 参数赋值给实例变量 self.appearance_thresh 用于指定外观相似性 阈值

        # 如果 args 中 with_reid 为 True ，则执行下面代码
        if args.with_reid:
            # 添加注意力机制试试效果
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        # 初始化一个GMC 对象，用于进行全局运动补偿。它接受args中cmc_method、 name 和 ablation 参数
        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        self.kcf_trackers = {}  # 初始化一个字典，用于存储KCF跟踪器


        # 添加显示 轨迹和 检测框函数
    def update(self, output_results, img):
      #  print("特征图维度14 up:", img.shape)  #  (540, 960, 3) 方法中的 img 图像都是 [w,h,c] 正常图像
      # ROT-Sort 方法中 img 图像类都是 [w,h,c] 正常图像，应该怎么处理这样的图像，能够增强，连续帧之间目标的关联性

        self.frame_id += 1  # 将帧计数器加1，用于跟踪当前处理的帧数
        activated_starcks = [] # 初始化一个空列表，用于存储激活的轨迹（track）
        refind_stracks = [] # 。。用于存储重新定位的轨迹
        lost_stracks = []  # 用于存储丢失的轨迹  # ？？？其中在捞一下，是否有丢失从显的
        removed_stracks = [] # 用于存储移除的轨迹，


        # 判断输出结果 output_result 是否为空
        if len(output_results):
            # 如果非空，并且输出结果的形状的第二维为5 ，表示输出结果包含置信度和边界框信息
            if output_results.shape[1] == 5:
                scores = output_results[:, 4] # 提取置信度信息？？
                bboxes = output_results[:, :4] # 提取边界框信息
                classes = output_results[:, -1] # 以及类别信息
            else:
                # 输出包含置信度和类别的概率
                scores = output_results[:, 4] * output_results[:, 5] # 将置信度和类别概率相乘得到最终置信度
                bboxes = output_results[:, :4]  # x1y1x2y2   # 提取边界框信息
                classes = output_results[:, -1] # 提取类别信息

            # Remove bad detections  # 从输出结果中移除 置信度 低于 设定阈值 self.track_low_thresh 的检测结果
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections  # 从筛选后的结果中选择 置信度 高于 设定阈值 args.track_high_thresh 的检测结果
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds] # 提取保留的边界框信息
            scores_keep = scores[remain_inds]  # 提取保留的置信度信息
            classes_keep = classes[remain_inds] # 提取保留的类别信息

        else:
            # 输出结果为空，初始化空列表
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings  提取嵌入特征 '''    # 第10行加入 embeding feature  嵌入 特征
      # 如果 self.args.with_reid 为真，则调用 self.encoder.inference  方法
      # 传入 img和边界框 dets ，获取对应的嵌入特征 feature_keep . key1
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)
            # print("特征图大小13 up:", img.shape)  # (540, 960, 3)
            # print("特征图大小16- up:", features_keep.shape)  # (5, 2048)


        # 如果dets 非空 ，启用嵌入特征提取 进行处理
        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                # 根据边界框dets, 置信度scores_keep, 嵌入特征features_keep  创建 STrack 对象将其存储在列表detections 中
                # key2
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                # 如果未启用 嵌入特增提取，根据边界框 dets 和 置信度 scores_keep 创建STrack 对象，并将其存储在列表detections中
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []  # 如果dets 为空，则将detections 初始化为空列表

        ''' Add newly detected tracklets to tracked_stracks''' # 增加新检测的轨迹 tracklets，添加到已跟踪轨迹列表tracked_stracks中
        unconfirmed = [] # 初始化一个空列表，用于存储未激活的轨迹 （既还未被确认的轨迹）
        tracked_stracks = []  # type: list[STrack]  # 存储已经激活 或其它状态 的轨迹
        # 遍历已跟踪的轨迹列表 self.tracked_stracks
        for track in self.tracked_stracks:

            # print("track1 {}".format(track))

            # 如果轨迹未被激活 （is_activated 为假），将其添加到unconfirmed 列表中
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                # 否者将其添加到 tracked_stracks
                tracked_stracks.append(track)
                # print("track2 {}".format(track))
                # print("track2 {}".format(track))

        ''' Step 2: First association, with high score detection boxes''' # 使用置信的较高的检测框进行关联
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks) # 将已经跟踪的轨迹和丢失的轨迹合并为一个轨迹池

        # Predict the current location with KF
        STrack.multi_predict(strack_pool) # 使用卡尔曼滤波器预测轨迹当前位置。 ？？ 这个地方可以加 注意力？ 或加其它弱线索？？

        # 利用相机校正技术对，Kalman Filter 的参数矩阵进行调整，意味着使用相机校正技术来优化Kalman Filter 的参数矩阵
        # Fix camera motion
        # 得到转换矩阵  key3 使用相机运动补偿来调整 卡尔曼滤波器的参数
        warp = self.gmc.apply(img, dets)

        # print("特征图维度15 up:", img.shape) #  (540, 960, 3)

        # 得到转换矩阵 warp 也是关键一步，wrap 矩阵的求解有多种办法，不过default 方法稀疏光影 sparseOptFlow
        # 调整 strack_pool 和 nconfirmed 中所有tracklet 的 kalman filter 参数
        # 使用相机运动补偿技术，来调整 strack_pool 和 unconfirmed 中 所有轨迹的卡尔曼滤波器参数
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # for detection in detections:
        #     i=1
        #     print("detection %d", i)
        #     print("detection {}".format(detection))

        # 显示相应的轨迹 和 检测 框框

        # Associate with high score detection boxes 使用高置信度的检测框进行轨迹关联
        # 首先用 IOU 距离进行筛选，大于一定阈值 直接配对 失败？？。。
        # 当然，计算IOU距离时候，也可以加入detection 的置信度分数


        ious_dists = matching.iou_distance(strack_pool, detections)  # 计算轨迹与 检测框 之间的iou距离
        # ious_dists = matching.iou_distance1(strack_pool, detections)  #  计算轨迹与 检测框  之间的giou距离 测试 效果直接为快下降为0

        # 论文上说， 马氏距离 计算相似度 与 贪婪算法 更匹配
        # trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.tracked_stracks]
        #
        # ious_dists = matching.compute_m_distance(detections,strack_pool, trk_innovation_matrix)
        #
        # print("ious1",ious_dists)

        ious_dists_mask = (ious_dists > self.proximity_thresh)  #？？？

        # 合并 IOU距离 和 检测框  的置信度分数 （对于MOT20 数据集不适用）
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
#  第18行，在第一次分配，计算 tracks 和 detection的 距离时，除了考虑IOU 距离，还会考虑 embeding feature 之间的距离
        # 如果启用了 reid 功能，基数按轨迹 与检测框之间的 嵌入特征距离
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0   # 计算嵌入特征距离的余弦距离并 除以2

            # emb_dists = matching.gate_cost_matrix(self.kalman_filter, emb_dists,strack_pool, detections)  # 没有掩码，试试门控处理，是否能优化矩阵

            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0   # 使用外观阈值 过滤 距离较大的轨迹
            emb_dists[ious_dists_mask] = 1.0   # 使用iou掩码过滤距离较远的轨迹
            dists = np.minimum(ious_dists, emb_dists)     # 从 iou距离 和嵌入特征距离，取二者的最小值 为最终距离

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists   # 没有启用reid ，直接用iou距离
        # 使用 线性分配算法（匈牙利算法） 将轨迹与检测框 进行关联。    dists信息中 这个里面既包含轨迹也包含检测框信息 ？？

        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # matches, u_track1, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)


        # online_im = plot_tracking12356(img, matches, u_track, u_detection )
        #
        #     # 显示图像 （显示绘制的图像）
        # cv2.imshow('Tracking Results1', online_im)
        #     # 等待键盘输入，键盘输入0 时关闭当前帧，显示下一帧跟踪结果
        # cv2.waitKey(0)
        #     # # 关闭图像窗口
        # cv2.destroyAllWindows()

        # matches, u_track, u_detection = other_matching_1.GreedyAssignment(dists, threshold=0.8,method='global')   # threshold 阈值在track 中 为 0.8
        # print("score:",detections[0].score)
      # 假设使用第一个检测框的分数计算阈值

        if len(detections) > 0:

            threshold1 = STrack.adaptive_threshold(detections[0].score)   # 第三次阈值设置，也和参考论文上相似，使用高阈值 0.7-0.99 之间
            # print("detections[0] 3.score",detections[0].score)
        else:
            threshold1 = 0.7

        # threshold1 = STrack.adaptive_threshold(detections[0].score)  # 使用第一个检测框 的分数对阈值在一定范围内扰动
        matches, u_track, u_detection = other_matching_1.GreedyAssignment(dists, threshold=threshold1, method='global')  # 调整阈值，为0.7， 寻找合适阈值 0.7比0.8 好效果



        # 初始化4个列表分别存储匹配成功的，检测框信息、轨迹框信息。没有匹配上的高分检测框，以及轨迹,高分检测框分数,第三次匹配的检测框
        matche_t1 =[]
        matche_d1 =[]
        matche_tid = []
        um_track =[]
        un_det   = []
        un_det_score = []
        T_match_d = []
        T_match_t = []

        # 处理匹配的轨迹和检测框
        for itracked, idet in matches:
            track = strack_pool[itracked]  #  itracked 代表轨迹在轨迹池中的索引
            det = detections[idet] # idet 代表检测框在检测结果中的索引
            if track.state == TrackState.Tracked:
                # 如果轨迹已被跟踪，则更新其状态和信息
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 如果轨迹未被跟踪，则重新激活并使用检测框信息
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            matche_t1.append(track.tlwh)   # 已经匹配的轨迹框 和 检测框,匹配轨迹的id
            matche_tid.append(track.track_id)
            matche_d1.append(det.tlwh)

        # 未匹配的轨迹框 和 高分检测框 用于可视化
        if len(u_track1) > 0:
            for T in u_track1:
                u1_track = strack_pool[T]  # 'self.track_pool' 假设是轨迹池
                um_track.append(u1_track.tlwh)

        if len(u_detection) > 0:
            for D in u_detection:
                d1_det = detections[D]  # 'self.detections' 假设是检测结果
                un_det.append(d1_det.tlwh)
                un_det_score.append(d1_det.score)


        ''' Step 3: Second association, with low score detection boxes'''  # 第二次关联随着低分检测框
      # 如果存在检测框的置信度分数
        if len(scores):
            # 根据置信度分数阈值，将检测框分为 高置信度 和 低置信度
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)

            # 根据低置信度筛选的 检测框和相应的分数、类别
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # 创建与低置信度检测框对应的辅助轨迹
        # association the untrack to the low score detections  将未跟踪和 低分检测相关联
        if len(dets_second) > 0:
            '''Detections'''  # key1
            # 将 低置信度的检测框转换为 辅助轨迹对象
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # 获取当前任被跟踪的 轨迹
        # # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        #
        r_tracked_stracks = [strack_pool[i] for i in u_track1 if strack_pool[i].state == TrackState.Tracked]   # 修改第一次未匹配的轨迹

        # r_tracked_stracks = [strack_pool[i] for i in u_track1 if strack_pool[i].state == TrackState.Tracked]   # 修改u_track 为u_track1 让第三次匹配使用这个第一次剩余的轨迹
        dists = matching.iou_distance(r_tracked_stracks, detections_second)  # 计算 仍被跟踪轨迹 与 低置信度检测框 之间的 iou 距离

        # dists = matching.iou_distance1(r_tracked_stracks, detections_second)  # 尝试在 低置信度上 使用 Giou，结果没有iou效果好，idf1指标稍微上升，但MOTA大量下降

      # # 使用线性分配算法（除了这个关联方法，还有它更好的代替？） 将 任被跟踪的轨迹 与低置信度检测框进行关联
      #  matches 匹配结果, u_track 未匹配的轨迹 索引, u_detection_second 未匹配 低检测框 索引


        # 这样扰动，感觉将阈值调到了 0.6多到0.7多范围 试试效果  试了试没有直接用0.5 阈值，效果好
        # if len(detections) > 0:
        #     threshold2 = STrack.adaptive_threshold_1(detections[0].score)  # 第三次阈值设置，也和参考论文上相似，使用高阈值 0.7-0.99 之间
        #     # print("detections[0]1.score",detections[0].score)
        # else:
        #     threshold2 = 0.5

        # print("threshold2:", threshold2)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5) # 原论文 代码

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)


        # matches, u_track, u_detection = other_matching_1.GreedyAssignment(dists, threshold=0.55, method='global')  # 报错 显示458 行 索引超出范围
      # 处理匹配的 跟踪轨迹 和 低置信度 检测框  # 遍历匹配的结果 轨迹和低置信度检测框（每个结果是一对，包含轨迹索引 检测框索引）
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]    # 根据索引 获取当前匹配到的 轨迹
            det = detections_second[idet]          # 根据索引 获取当前匹配到的 低置信度检测框
            if track.state == TrackState.Tracked:
                # 如果轨迹已被跟踪，则更新其状态和信息
                track.update(det, self.frame_id)
                activated_starcks.append(track)  # 更新后的轨迹 添加到 已激活的轨迹列表中
            else:
                # 如果轨迹 未被跟踪，则重新激活并使用检测框信息
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)   # 更新后的轨迹 添加到 重新发现的 轨迹列表中


      #  Third association, with high score detection boxes and Gaussian distance' 第三次级联 随着高分检测框与 Gaussian distance
        use_motion =True
        if use_motion:
            # get the remain of the high score detection boxes 获取剩余高分检测框
                detections = [detections[i] for i in u_detection]

                # r_r_tracked_stacks = [r_tracked_stracks[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]  # 试图匹配激活状态轨迹

                # 它从一个可迭代对象 u_track 中提取元素，并将这些元素用作索引来访问 r_tracked_stracks 列表中的对应项，最终生成一个新的列表。
                r_r_tracked_stacks = [r_tracked_stracks[i] for i in u_track] # r_tracked_stracks 是已激活的轨迹集合

                # 将未激活的轨迹合并到激活状态的轨迹集合中，想法是扩充轨迹集合，将新出现的目标视为激活状态的轨迹，在第三次匹配时进行匹配
                # 因为新出现的目标通常视为 未激活状态，该目标往往是高分检查框 中的真实目标
                r_r_tracked_stacks.extend(unconfirmed)  # # 将未激活的轨迹合并到激活状态的轨迹集合中


                ious_dists3 = matching.iou_distance(r_r_tracked_stacks,  detections)

                matches, u_track, u_detection_second = matching.linear_assignment(ious_dists3, thresh=0.7)





                # 遍历每对 匹配的轨迹和检测框
                for itracked, idet in matches:
                    track = tracked_T[itracked] # 获取匹配的轨迹
                    det = detections[idet]  # 获取匹配的检测框
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id) # 更新轨迹   感觉可以把这两行直接拿出来，把else 中的内容 去掉 因为高分检测就可以理解为跟踪成功，在行人里面
                        activated_starcks.append(track) # 将轨迹 添加到activated_starcks
                    T_match_d.append(det.tlwh)  # 将第三次匹配成功的目标显示
                    T_match_t.append(track.tlwh) # 第三次匹配成功的轨迹


                r_tracked_stracks = r_r_tracked_stacks  # 跟新 r_tracked_stracks # 第三次匹配结束--


        # 标记未匹配到的轨迹为 丢失状态
        for it in u_track:
            # 遍历未匹配的 轨迹列表，索引信息 获取当前未匹的 轨迹
            # track = r_tracked_stracks[it]
            track = r_r_tracked_stacks[it]
            if not track.state == TrackState.Lost:
                # 如果轨迹状态不是丢失状态
                track.mark_lost()   # 将轨迹标记为丢失状态
                lost_stracks.append(track)  # 将标记为丢失状态的 轨迹 添加到丢失轨迹列表中

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''  # 处理未确认的轨迹，通常轨迹仅仅在 开始帧

        detections = [detections[i] for i in u_detection]  # 获取未确认 轨迹对应的检测
        ious_dists = matching.iou_distance(unconfirmed, detections)  # 计算未确认轨迹 与 检测框 之间的 iou 距离

        # # 由于修改上面，这边要做相应修改
        # if len(u_detection)>0:
        #     detections = [detections[i] for i in u_detection]     # 获取未确认 轨迹对应的检测
        #     ious_dists = matching.iou_distance(unconfirmed, detections)  # 计算未确认轨迹 与 检测框 之间的 iou 距离
        # else:
        #     ious_dists = matching.iou_distance(unconfirmed, u_detection)  # 如果 u_detection 中没有内容，直接使用 上一个 detections 比较？

      # ious_dists = matching.v_iou_distance(unconfirmed, detections)  # 计算未确认轨迹 与 检测框 之间的 iou 距离

        ious_dists_mask = (ious_dists > self.proximity_thresh)   # 根据距离是否超过阈值  来创建掩码
        if not self.args.mot20:   # 如果不是MOT20 数据集 则使用融合得分
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:    # 如果使用 reid 特征
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0 # 计算未确认轨迹 与检测框 之间的 Reid 距离 并 除以2
            raw_emb_dists = emb_dists.copy()   # 复制reid 距离，用于备份
            emb_dists[emb_dists > self.appearance_thresh] = 1.0  # 将 reid 特征距离大于 self.appearance_thresh 阈值部分置为1
            emb_dists[ious_dists_mask] = 1.0  # 将与未确认 轨迹有较近 iou 距离 的检测框的 reid 特征距离设置为1
            dists = np.minimum(ious_dists, emb_dists)   # 将iou 距离 和 reid 特征距离取最小 作为 最终的距离矩阵
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)  # 使用线性分配算法进行匹配
        # matches, u_unconfirmed, u_detection = other_matching_1.GreedyAssignment(dists, threshold=0.7, method='global')
        for itracked, idet in matches:   # 更新匹配到的   轨迹  与 检测框 的匹配对？？ 遍历成功配对的 检测框 和  轨迹
            unconfirmed[itracked].update(detections[idet], self.frame_id)  # 使用匹配到的检测框 更新未确认 轨迹的状态和信息，同时更新帧 id
            activated_starcks.append(unconfirmed[itracked]) # 将更新后 的未确认轨迹 添加到已激活轨迹列表中

            # T_match_d.append(detections[idet].tlwh)  # 将第三次匹配成功的目标显示  实际对未确认轨迹和 高分检测进行匹配
            # T_match_t.append(unconfirmed[itracked].tlwh)  # 第三次匹配成功的轨迹

        for it in u_unconfirmed:  # 遍历未匹配到的 未确认轨迹为
            track = unconfirmed[it] # 获取当前未匹配到的 未确认轨迹
            track.mark_removed() # 将未匹配到的 未确认轨迹标记为移除状态
            removed_stracks.append(track) # 将标记为 移除状态的 未确认轨迹添加到移除 轨迹列表中

        # # 调用函数并绘制结果  第三次匹配成功为紫色
        # online_im = plot_tracking_custom1(img, matche_d1, um_track, un_det, un_det_score,T_match_d,T_match_t,frame_id=self.frame_id)
        # # 显示图像
        # cv2.imshow('Tracking Results T', online_im)
        # cv2.waitKey(0)  # 等待键盘输入
        # cv2.destroyAllWindows()  # 关闭图像窗口




        """ Step 4: Init new stracks""" # 初始化新轨迹
        for inew in u_detection:    # 遍历失配的 检测框列表
            track = detections[inew]  # 根据索引获取当前 失配的检测框
            if track.score < self.new_track_thresh:   # 如果检测框的 得分小于新轨迹阈值，则跳过该检测框
                continue

            track.activate(self.kalman_filter, self.frame_id)   # 激活新的轨迹，使用卡尔曼滤波器和 当前帧id
            activated_starcks.append(track)  # 将激活的新轨迹 添加到已激活的轨迹列表中

        """ Step 5: Update state"""  # 跟新轨迹状态
        for track in self.lost_stracks:   # 遍历已丢失的轨迹列表
            if self.frame_id - track.end_frame > self.max_time_lost:    # 如果当前帧id 减去轨迹 结束帧id 大于 最大丢失时间
                track.mark_removed()  # 标记轨迹为移除状态
                removed_stracks.append(track) # 将标记为移除状态的轨迹添加到 移除轨迹列表中

        """ Merge """ # 合并操作
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]  # 过滤掉状态不是 “Tracked”的 已跟踪轨迹，只保留状态为 Tracked的 轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks) # 将已跟踪轨迹列表与 激活的新轨迹列表合并
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks) # 将合并后的已跟踪轨迹列表 与 重新定位的轨迹列表合并
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks) # 从已丢失轨迹列表中 移除 已跟踪轨迹列表中 的轨迹
        self.lost_stracks.extend(lost_stracks)  # 将新丢失的轨迹列表 添加到已丢失轨迹列表中
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)  # 从已丢失的轨迹列表中 移除 已移除轨迹列表中 存在的轨迹 ？？
        self.removed_stracks.extend(removed_stracks)   # 将新移除的轨迹列表 添加到 已移除轨迹 列表中
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)  # 移除 已跟踪轨迹列表和 已丢失轨迹列表中的重复轨迹 ？

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks] # 生成输出轨迹列表，只包含已激活的已跟踪轨迹

        # self.pred_bboxes = np.array([x.tmp_pred for x in self.tracked_stracks if hasattr(x, "tmp_pred")])   # 添加临时预测

        return output_stracks # 返回输出轨迹列表作为多目标跟踪的结果


def joint_stracks(tlista, tlistb):
    exists = {}  # 创建一个空字典 exists 用于存储已存在的轨迹id
    res = [] # 创建一个空列表res 用于存储结果轨迹列表
    for t in tlista:   # 遍历tlista 中每个轨迹t，将其轨迹 id， track_id 添加到 exists 字典中作为 键，并将值设为1，表示该轨迹id 已存在
        exists[t.track_id] = 1
        res.append(t) # 将轨迹t 添加到结果列表 res 中
    for t in tlistb:    # 遍历 tlistb 中每个轨迹t ，获取其轨迹id，t.track_id 。
        tid = t.track_id
        if not exists.get(tid, 0):  # 如果exists 字典中 不存在该轨迹id，则将其添加到 exists 字典中，并将该轨迹t 添加到结果列表res中
            exists[tid] = 1
            res.append(t)
    return res

# 该函数的功能是从 tlista 中 删除 tlistb 中已存在的 轨迹对象，并返回剩余的轨迹对象
def sub_stracks(tlista, tlistb):
    stracks = {}  # 创建一个空字典用于存储 轨迹对象
    for t in tlista:   # 遍历tlista 中 的每个轨迹对象
        stracks[t.track_id] = t   # 将轨迹对象存储到字典 stracks中 ，以轨迹id作为键
    for t in tlistb:  # 遍历tlistb中 的每个轨迹对象
        tid = t.track_id  # 获取轨迹id
        if stracks.get(tid, 0):  # 检查指点stracks 中是否存在具有相同轨迹id的 轨迹对象
            del stracks[tid]     # 如果存在，则从字典中删除该轨迹对象
    return list(stracks.values()) # 将字典strack中剩余的轨迹对象转换为 列表并返回列表

# 函数功能是 从 stracksa, stracksb 两组轨迹中删除 相互之间重复的轨迹
def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)  # 计算两组轨迹之间的iou矩阵
    # pdist = matching.v_iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)  # 找到距离小于 0.15 的轨迹对的索引
    dupa, dupb = list(), list()  # 创建两个空列表用于 存储 重复的轨迹索引
    for p, q in zip(*pairs):   # 遍历距离小于 0.15 的轨迹对 的索引
        timep = stracksa[p].frame_id - stracksa[p].start_frame   # 计算轨迹a的时长
        timeq = stracksb[q].frame_id - stracksb[q].start_frame   # 计算轨迹b的时长
        if timep > timeq:    # 如果轨迹a的时长 大于轨迹b的时长
            dupb.append(q)    # 将轨迹b的索引 添加到重复的 轨迹b 列表中
        else:                # 如果轨迹 a的时长小于 等于轨迹b的时长
            dupa.append(p)   # 将轨迹a的 索引添加到 重复的 轨迹b 列表中
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]   # 从stracksa 中删除重复的轨迹对象
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]    # 从stracksb 中删除重复的轨迹对象
    return resa, resb   # 返回删除重复的轨迹后的结果列表 resa和 resb
