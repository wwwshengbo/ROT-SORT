import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

# 把yolox 换成 yolov8 ,改进yolov8
from yolox.data.data_augment import preproc   # # 导入数据预处理模块，用于对输入数据进行增强和处理，以提高模型鲁棒性。可能包含图像变换，归一化、裁剪、缩放等操作
from yolox.exp import get_exp   # 实验配置模块，用于获取YOLOX  模型的实验配置，他提供一组配置选项，如模型结构、训练超参数、数据集路径等，可根据自己需要定制
#  fuse_model 这个模块用于模型融合，它将模型的一些层或操作合并为一个更高效的层或操作，以加速推理过程。
#  get_model_info 模块用于 获取模型的信息，如模型的输入尺寸，输出尺寸、参数数量等。帮助用于了解模型的结构和规模，并进行模型选择和调优
#  postprocess 改模块用于后处理，将模型的输出结果进行解码和处理，以得到最终的目标检测结果或跟踪结果。它可能包含非极大值抑制（NMS）、目标解码，结果过滤等操作
from yolox.utils import fuse_model, get_model_info, postprocess  # 分别在 262 、246、143 行
from yolox.utils.visualize import plot_tracking # 用于可视化多目标跟踪的结果。它除了可以绘制跟踪框、标签、轨迹等信息在图像或视频上，以便直观地观察和分析跟踪效果  197 行

from tracker.tracking_utils.timer import Timer


from tracker.rot_sort import ROTSORT

from GSI import GSInterpolation
from other_interpolation import GSInterpolation1  # 自己修改的 高斯插值

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("ROT-SORT Tracks For Evaluation!")

    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    # parser.add_argument("--eval", dest="split_to_eval", type=str, default='train', help="split to evaluate: train | val | test")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='val',help="split to evaluate: train | val | test")
    # 如果将上面的 default 设置为 val 既进行验证模式 既训练集每个序列的一半
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    # parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keep lost tracks") # 提高轨迹的连续性和完整性
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    # parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser

# 获取图像列表
def get_image_list(path):
    image_names = []   # 创建一个空 列表用于存储图像文件路径
    # os.walk(path) 返回一个迭代器，遍历指定路径下的所有文件夹和文件。它返回一个元组，包含当前文件夹路径、当前文件夹下的子文件夹列表和当前文件夹下的文件列表
    for maindir, subdir, file_name_list in os.walk(path):  # 遍历指定路径下的所有文件 和文件夹
        for filename in file_name_list:  # 遍历当前文件夹下的 所有文件
            # 构建每个文件路径 通过使用 通过使用 osp.join(maindir, filename)，其中 osp 是 os.path 模块的别名。
            apath = osp.join(maindir, filename)   # 构建文件的绝对路径
            ext = osp.splitext(apath)[1]          # 获取文件的扩展名   osp.splitext(apath)将文件路径 分割为 文件名 和扩展名 的元组 [1] 表示取元组中的第二个元素，既文件扩展名
            if ext in IMAGE_EXT:                  # 检查文件扩展名是否在指定 的图像扩展名列表中
                image_names.append(apath)         # 如果是图像文件，则将文件路径添加到列表中
    return image_names                            # 返回文件路径列表

# 保存跟踪结果
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

# 定义一个 Predictor 的类，用于进行目标检测的预测操作，在初始化方法 __init__中以下变量被赋值
class Predictor(object):
    def __init__(
            self,
            model,   # 目标检测模型
            exp,      # 包含实验配置的对象
            device=torch.device("cpu"),   # 指定模型在那个设备上运行，默认为 CPU
            fp16=False      #是否使用半精度浮点数运算
    ):
        self.model = model        # 初始化模型
        self.num_classes = exp.num_classes  # 类别数量
        self.confthre = exp.test_conf    # 置信度阈值
        self.nmsthre = exp.nmsthre       # NMS(非极大值抑制)阈值
        self.test_size = exp.test_size   # 测试图像的尺寸
        self.device = device             # 设备
        self.fp16 = fp16                 # 是否使用 半精度 浮点数计算

        self.rgb_means = (0.485, 0.456, 0.406)  # 图像均值 (用于归一化) rgb_means 包含三个值，分别表示 图像的红、绿、蓝三通道均值
        self.std = (0.229, 0.224, 0.225)        # 图像标准差(用于归一化) std 同样包含三个值，分别表示图像 红、绿、蓝 三通道的标准差

# inference 方法，用于 进行目标检测的 推理操作。该方法接受图像和计时器，作为输入，并返回推理结果和图像信息字典
    def inference(self, img, timer):
        img_info = {"id": 0}    # 创建图像信息字典
        # 检查输入的图像是否为字符串(图像文件路径)
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)   # 获取图像文件名
            img = cv2.imread(img)     # 读取图像文件
        else:
            img_info["file_name"] = None
        # 检查图像是否为空
        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        # 获取图像的高度和 宽度
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img  # 保存原始图像(未经处理)

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std) # 对图像进行预处理（尺寸调整，归一化等） key 3
        img_info["ratio"] = ratio   # 保存尺寸调整比例
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)  # 转换为张量并移动到指定设备
        if self.fp16:
            img = img.half()  # to FP16  # 转换为 半精度浮点数(如果设置为 FP16 计算)
        # 禁用梯度计算，进行模型推理
        with torch.no_grad():
            timer.tic()   # 计时器开始计时
            outputs = self.model(img)   # 模型推理得到输出结果 ？？？这个模型是检测模型 yolox？？还是使用
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)     # 后处理结果 (筛选 NMS等)  key 4

        return outputs, img_info  # 返回推理结果 和 图像信息字典

# 用于 图像跟踪
def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):         # 检查输入路径是否为 目录
        files = get_image_list(args.path)  # 获取目录中的文件列表
    else:
        files = [args.path]   # 否者将输入路径作为单个图像文件
    files.sort()  # 对图像文件列表进行排序 以确保按照顺序进行跟踪
    # 如果启用了 消融模式(ablation)，则从中间位置开始跟踪
    if args.ablation:
        files = files[len(files) // 2 + 1:]  # 从文件列表的中间位置开始跟踪。为了在消融实验中排出一部分图像

    num_frames = len(files)  # 图像帧数 既文件列表的长度

    # Tracker   创建跟踪器对象 传入 args 参数 和 帧率 args.fps
    tracker = ROTSORT(args, frame_rate=args.fps)

    results = []   # 用于保存跟踪结果的列表

    for frame_id, img_path in enumerate(files, 1):
        # 遍历图像文件列表，使用 enumerate 函数获取帧id 和图像路径
        # Detect objects 检测目标对象
        outputs, img_info = predictor.inference(img_path, timer)  # 调用 predictor.inference 方法进行目标检测   key 5
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))  # 计算缩放比例，用于将检测框坐标缩放到原始图像中，是否可在次操作？？？

        if outputs[0] is not None:       # 如果存在目标检测结果
            outputs = outputs[0].cpu().numpy()     # 将输出转化为 numpy 数组
            detections = outputs[:, :7]     # 提取检测框信息（前7个元素）
            detections[:, :4] /= scale     # 根据缩放比例，缩放检测框坐标

            trackerTimer.tic()  #  启动跟踪器计时器
            online_targets = tracker.update(detections, img_info["raw_img"])  #调用跟踪器的 update 方法，传递检测结果 和 原始图像 信息   key 6
            trackerTimer.toc()   # 停止跟踪器计时器，并记录经过的时间
            # 初始化几个空列表，用于保存在线目标跟踪结果的列表
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                # 遍历在线目标跟踪结果
                tlwh = t.tlwh   # 获取目标的位置信息(top left x, top left y, width, height)
                tid = t.track_id   # 获取，目标的跟踪id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh   # 判断目标的纵横比是否大于 阈值
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:   # 如果目标框的面积大于 最小框面积阈值，并且纵横比小于等于阈值
                    online_tlwhs.append(tlwh) # 将目标位置信息添加到列表中
                    online_ids.append(tid) # 将目标跟踪id 添加到列表中
                    online_scores.append(t.score) # 将目标的置信度 添加到列表中

                    # save results  保存结果，将结果以特定格式(CSV 行)添加到结果列表中
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc() # 停止计时器，并记录记过的时间
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
            # 调用  plot_tracking 函数绘制跟踪结果并 返回图像
        else:
            timer.toc()
            online_im = img_info['raw_img']  #  如果没有目标检测结果，直接使用原始图像 作为输出图像

        # cv2.imshow("named",online_im) # 添加显示 ，该显示，当按下关键时，显示退出
        # 下面一行代码是控制显示速度，当 waitKey 内数值为 1000时 每秒显示1帧跟踪结果
        # ch = cv2.waitKey(1000)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):

        # 如果设置了保存帧的标志
        if args.save_frames:
            save_folder = osp.join(vis_folder, args.name)  # 构建保存帧的文件夹路径
            os.makedirs(save_folder, exist_ok=True)    # 创建保存帧的文件夹，如果文件夹已存在 则忽略
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)   # 将当前帧的 图像保存到指定文件夹中

        if frame_id % 20 == 0:   # 每20帧执行一次以下操作
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time))) # 打印当前处理的帧数，总帧率和平均帧率

    res_file = osp.join(vis_folder, args.name + ".txt")   # 构建结果文件夹路径
# 打开结果文件 已写入模式
    with open(res_file, 'w') as f:
        f.writelines(results)    # 将结果列表中的内容，写入文件中
    logger.info(f"save results to {res_file}")   # 打印保存结果的路径

    GSInterpolation1(path_in=res_file, path_out=res_file, interval=10, tau=10)  # 对跟踪结果使用插值 原论文相比只添加这一行其它都没变

def main(exp, args):
    # 如果没有提供 实验名称 ，则使用  exp.exp_name 作为默认实验名称
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # 创建输出目录，用于储存实验结果
    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    # 创建可视化结果的文件夹路径，用于跟踪结果
    vis_folder = osp.join(output_dir, "track_results_MOT17_train_GSI_1")  # 输出结果的 文件夹路径


    os.makedirs(vis_folder, exist_ok=True)
    # 设备选择，如果有gpu 就选 ，没有就选 cpu
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))   # 打印参数信息

    if args.conf is not None:
        exp.test_conf = args.conf    # 如果指定了args.conf参数 则将其作为 exp.test_conf 参数
    if args.nms is not None:
        exp.nmsthre = args.nms  # 如果指定了args.nms 参数 则将其作为 exp.nmsthre 参数
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)  # 如果指定测试时，大小 则使用，这个大小

    model = exp.get_model().to(args.device)   # 从exp 对象中获取模型，并移动到指定设备上
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))    # 打印模型的摘要信息，包括模型的结构和参数量
    model.eval()  # 设置为评估模式，既禁用训练模式中的一些 特定操作，如随机失活

    # 如果未提供 args.ckpt（checkpoint）参数，则设置 ckpt_file 为输出目录中的最佳模型权重文件路径
    if args.ckpt is None:
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt   # 如果提供了 args.ckpt 参数，则将其设置为 ckpt_file
    logger.info("loading checkpoint")  # 打印日志，表示正在加载检查点文件
    ckpt = torch.load(ckpt_file, map_location="cpu")  # 使用 torch.load 加载检查点文件，map_location="cpu" 表示将模型加载到 CPU 上

    # load the model state dict
    model.load_state_dict(ckpt["model"])  # 加载模型的状态字典（权重参数）
    logger.info("loaded checkpoint done.")  # 打印日志，表示检查点文件加载完成

    if args.fuse:   # 如果 args.fuse 参数为 True，则进行模型融合操作
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16   # 如果 args.fp16 参数为 True，则将模型转换为半精度浮点数格式（FP16）
    # 创建 Predictor 对象，用于进行预测
    predictor = Predictor(model, exp, args.device, args.fp16)   # key 1
    # 调用 image_track 函数，进行图像跟踪
    # image_track(predictor, vis_folder, args)
    res_file = image_track(predictor, vis_folder, args)               #key 2
    return res_file

if __name__ == "__main__":
    args = make_parser().parse_args()
    # 解析命令行参数，并将结果赋值给 args 变量
    data_path = args.path
    fp16 = args.fp16
    device = args.device
    # 从 args 中获取 数据路径、fp16标志 和设备信息，并分别赋值给相应的变量

    if args.benchmark == 'MOT20':
        # train_seqs = [1, 2, 3, 5]   # 原论文
        # train_seqs = [1, 2]
        train_seqs = [1, 2,3] # 由于内存溢出，前面两个序列跑完，再跑后面两个 or one
        test_seqs = [4, 6, 7, 8]
        # test_seqs = [8]
        seqs_ext = ['']
        MOT = 20
        # 如果 benchmark 参数为 MOT20 ，设置训练序列、测试序列、序列扩展 和 MOT 值
    elif args.benchmark == 'MOT17':
        # train_seqs = [2, 4, 5, 9, 10, 11, 13]
        train_seqs = [2, 4, 5,9]  # 验证时，可以随机取出几个
        # train_seqs = [11, 13]
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['FRCNN','DPM','SDP']    # 每个字符串代表一种不同的序列扩展方法或算法。这样的赋值表明 seqs_ext 变量被用于存储具体的扩展信息，用于区分和标识不同的序列扩展方式
        # seqs_ext = ['SDP']
        MOT = 17
        # # 如果 benchmark 参数为 MOT17 ，设置训练序列、测试序列、序列扩展 和 MOT 值
    elif args.benchmark == 'MOT16':
        # train_seqs = [2, 4, 5, 9, 10, 11, 13]
        train_seqs = [2, 4, 5, 9, 10, 11, 13]  # 验证时，可以随机取出几个
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['']  # 每个字符串代表一种不同的序列扩展方法或算法。这样的赋值表明 seqs_ext 变量被用于存储具体的扩展信息，用于区分和标识不同的序列扩展方式
        MOT = 16
        # # 如果 benchmark 参数为 MOT17 ，设置训练序列、测试序列、序列扩展 和 MOT 值
    else:
        raise ValueError("Error: Unsupported benchmark:" + args.benchmark)
    #  # 如果 benchmark 参数既不是 'MOT20' 也不是 'MOT17'，则抛出 ValueError 异常

    ablation = False      # 是否进行消融研究， 验证每个模块的性能
    if args.split_to_eval == 'train':
        seqs = train_seqs             # 如果 split_to_eval 参数为 train ，设置要评估的序列为 训练序列
    elif args.split_to_eval == 'val':
        seqs = train_seqs
        ablation = True                 # 如果 split_to_eval 参数为 'val'，设置要评估的序列为训练序列，并将 ablation 标志设置为 True
    elif args.split_to_eval == 'test':
        seqs = test_seqs                 # 如果 split_to_eval 参数为 'test'，设置要评估的序列为测试序列
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + args.split_to_eval)  # 如果 split_to_eval 参数既不是 'train' 也不是 'val' 也不是 'test'，则抛出 ValueError 异常

    mainTimer = Timer()   # 创建一个 名为 mainTimer 的 Timer 对象。 既实例化一个对象，并将 该对象赋值给 mainTimer，可以通过使用mainTimer 来调用 Timer 类中定义的方法
    mainTimer.tic()       # tic() 方法用于启动 计时器，既 记录当前时间作为 起始时间点。通过调用该方法可以测量某代码块 或操作的执行时间。
    # 该方法与 toc是成对出现的，toc() 用于停止计时器 并返回 经过的时间

    for ext in seqs_ext:
        # 对于 seqs_ext 列表中的每个元素，依次赋值给变量 ext
        for i in seqs:   # 对于 seqs 列表中的每个元素，依次赋值给变量 1
            if i < 10:
                seq = 'MOT' + str(MOT) + '-0' + str(i)
            else:
                seq = 'MOT' + str(MOT) + '-' + str(i)
            # 构建 seq 字符串，根据 序列编号 i 和 MOT 值拼接而成，如果i 小于10 ，则在序列编号前 补 零
            if ext != '':
                seq += '-' + ext
            # 如果 ext 不为空 字符串，则在seq 字符串末尾拼接上 ext
            args.name = seq   # 将seq 赋值给 args 的 name 属性

            args.ablation = ablation
            args.mot20 = MOT == 20
            args.fps = 30
            args.device = device
            args.fp16 = fp16
            args.batch_size = 1
            args.trt = False
            # 将各个超参数赋值给 args 对象的相应属性
            split = 'train' if i in train_seqs else 'test'   # 如果 i 在 train_seqs 列表中，则split 被赋值为  train  ，否则 为 test
            args.path = data_path + '/' + split + '/' + seq + '/' + 'img1'  # 构建数据路径，拼接data_path 、split、seq 和 img1 字符串

            # 输出注释？？
            # print(f"ext: {ext}, i: {i}, seq: {seq}, split: {split}, args.path: {args.path}")

            # 如果 args.default_parameters 为True ，则执行下面代码
            if args.default_parameters:

                if MOT == 20:  # MOT20
                    args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py'
                    args.ckpt = r'./pretrained/bytetrack_x_mot20.tar'
                    args.match_thresh = 0.7
                else:  # MOT17
                    if ablation:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_ablation.py'
                        args.ckpt = r'./pretrained/bytetrack_ablation.pth.tar'
                    else:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_det.py'
                        args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'

                exp = get_exp(args.exp_file, args.name)   # 获取实验配置 根据 args.exp_file, args.name

                args.track_high_thresh = 0.6               # 设置跟踪阈值 和缓冲区大小的 参数
                args.track_low_thresh = 0.1
                args.track_buffer = 30

                if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
                    args.track_buffer = 14
                elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
                    args.track_buffer = 25
                else:
                    args.track_buffer = 30
                # 根据 序列名称 seq 设置不同的 跟踪缓冲区大小
                if seq == 'MOT17-01-FRCNN':
                    args.track_high_thresh = 0.65
                elif seq == 'MOT17-06-FRCNN':
                    args.track_high_thresh = 0.65
                elif seq == 'MOT17-12-FRCNN':
                    args.track_high_thresh = 0.7
                elif seq == 'MOT17-14-FRCNN':
                    args.track_high_thresh = 0.67
                elif seq in ['MOT20-06', 'MOT20-08']:
                    args.track_high_thresh = 0.3
                    exp.test_size = (736, 1920)
                # 根据序列名称 seq 设置 不同的跟踪高阈值，并在某些情况下设置测试尺寸


                args.new_track_thresh = args.track_high_thresh + 0.1    # 根据跟踪高阈值 计算新的跟踪阈值
            else:
                exp = get_exp(args.exp_file, args.name)   # 获取实验配置 根据 args.exp_file, args.name

            exp.test_conf = max(0.001, args.track_low_thresh - 0.01)   # 设置置信度 配置参数，取  0.001, args.track_low_thresh - 0.01 中较大值
            main(exp, args)  # 调用main 函数 传递 exp 和  args 作为参数
            # res_file1 = main(exp, args)   # 调用main 函数 传递 exp 和  args 作为参数

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))

