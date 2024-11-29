import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

sys.path.append('.')

# 导入检测器用的包
# preproc 该函数在 146 行 为key4 重要函数
from yolox.data.data_augment import preproc   # 导入数据预处理模块，用于对输入数据进行增强和处理，以提高模型鲁棒性。可能包含图像变换，归一化、裁剪、缩放等操作
from yolox.exp import get_exp   # 实验配置模块，用于获取YOLOX  模型的实验配置，他提供一组配置选项，如模型结构、训练超参数、数据集路径等，可根据自己需要定制
#  fuse_model 这个模块用于模型融合，它将模型的一些层或操作合并为一个更高效的层或操作，以加速推理过程。
#  get_model_info 模块用于 获取模型的信息，如模型的输入尺寸，输出尺寸、参数数量等。帮助用于了解模型的结构和规模，并进行模型选择和调优
#  postprocess 改模块用于后处理，将模型的输出结果进行解码和处理，以得到最终的目标检测结果或跟踪结果。它可能包含非极大值抑制（NMS）、目标解码，结果过滤等操作
from yolox.utils import fuse_model, get_model_info, postprocess   # 分别在 363、346、163 行重要函数

# 用于可视化多目标跟踪的结果。它除了可以绘制跟踪框、标签、轨迹等信息在图像或视频上，以便直观地观察和分析跟踪效果
from yolox.utils.visualize import plot_tracking, plot_tracking121, plot_tracking12,plot_tracking123,plot_tracking1235,plot_trackingk, plot_trackingkTD


from tracker.rot_sort import ROTSORT


from tracker.tracking_utils.timer import Timer

import matplotlib.pyplot as plt
import matplotlib.patches as patches


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]





def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # NAS 根据置信度信息，自适应调整 KF 的 标准差
    parser.add_argument("--NSA", default="False", type=str, help="根据置信度信息，自适应调整 KF 的 标准差")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

#定义 Predictor 的类，用于目标跟踪的预测，在构造函数中，它接受 模型对象，实验配置对象和其它参数作为输入，并初始化一系列属性
# 如果提供 TensorRT 文件路径，他会加载并优化其它参数模型以加速推理，还定义RGB 通道的均值和标准差用于数据预处理
#
class Predictor(object):
    # 定义了 predictor 类构造函数，该构造函数接受以下参数
    def __init__(
        self,
        model, # 模型对象，用于进行预测
        exp,  # 实验配置对象，包含相关的配置参数
        trt_file=None,  # TensorRT 文件路径，用于加速推理。默认值为None ，表示不使用
        decoder=None,  # 解码器对象，用于解码模型的输出
        device=torch.device("cpu"),  # 指定模型计算设备，默认 CPU
        fp16=False  # 是否使用16位 浮点数进行推理，默认False
    ):
        self.model = model  # 将传入的 model 参数赋值给Predictor 类的model 属性
        self.decoder = decoder # 将传入的 decoder 参数赋值给Predictor 类的decoder 属性
        self.num_classes = exp.num_classes  # 从实验配置对象exp中 获取目标类别数量，并赋值给Predictor 类的num_classes 属性
        self.confthre = exp.test_conf # 从实验配置对象exp中 获取测试 置信度 阈值的值，并赋值给Predictor 类的confthre 属性
        self.nmsthre = exp.nmsthre # # 从实验配置对象exp中 获取非极大值抑制(NMS)阈值的值，并赋值给Predictor 类的nmsthre属性
        self.test_size = exp.test_size # 从实验配置对象exp中 获取测试尺寸的值，并赋值给Predictor 类的test_size属性
        self.device = device
        self.fp16 = fp16

        # 检查 trt_file 是否不为none
        if trt_file is not None:
            from torch2trt import TRTModule # 从torch2trt库中导入  TRTModule 类

            model_trt = TRTModule() # 创建一个TRTModule 对象用于加载 TensorRT 模型
            model_trt.load_state_dict(torch.load(trt_file)) # 从指定的 TensorRT 文件中加载模型的状态字典，并将其赋值给model_trt 对象

            # 创建一个大小为 (1, 3, exp.test_size[0], exp.test_size[1]) 大小的张量x， 并将其移动到指定设置上
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x) # 对模型 self.model 执行一次前向传播，以便创建 TensorRT 引擎。这行代码的目的是为了在模型上执行一次前向传播，从而创建 TensorRT 引擎并优化模型的推理性能。
            self.model = model_trt # 将加载的 TensorRT 模型赋值给 Predictor 类的 model 属性，以便后续使用优化后的模型进行推理。
        self.rgb_means = (0.485, 0.456, 0.406) #将RGB 通道的均值设置为(0.485, 0.456, 0.406) 用于数据预处理
        self.std = (0.229, 0.224, 0.225) # 将RGB 通道的标准差设置为 (0.229, 0.224, 0.225) 用于数据预处理

    # 该函数接受图像数据img 和 计时器对象 timer 作为输入
    def inference(self, img, timer):
        img_info = {"id": 0}  # 创建一个 字典 img_info，用于存储 图像信息。初始时，将图像的 ID 设置为 0
        if isinstance(img, str): # 检查 img 是否是字符串类型，判断是否输入的是图像文件路径
            #os.path.basename()函数会返回路径中的最后一部分，即文件名或目录名 例如 ：path = '/path/to/file.txt' filename = os.path.basename(path)
            img_info["file_name"] = osp.basename(img) # 将图像的基本名称（不包含路径信息）存储在 img_info 字典中的 “file_name”键中
            img = cv2.imread(img) # 如果 img 是图像文件路径，使用 imread 函数读取图像数据，并将其赋值给img 变量
        else:
            img_info["file_name"] = None # 如果img 不是字符串类型，将 None（作为 value） 存储在 img_info 字典中的 “file_name”键中

        height, width = img.shape[:2] # 获取图像的高度和宽度，并将其分别赋值 给 height 和 width 变量
        img_info["height"] = height # 将图像的高度 作为 value 存储在 img_info 字典中“height”键（key） 中
        img_info["width"] = width #  将图像的宽度 作为 value 存储在 img_info 字典中“width”键（key） 中
        img_info["raw_img"] = img # 将原始图像数据 存储 存储在 img_info 字典中的 raw_img 键中

        # 调用preproc 函数对图像进行预处理，包含尺寸调整，颜色标准化等，返回预处理后的图像数据 img 和 尺寸调整比例 ratio， 进入下面函数查看内部代码
        # print("特征图维度:12-",img.shape) #(540, 960, 3)

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)  # key4
        # print("特征图维度:--",img.shape)   # (3, 800, 1440)


        img_info["ratio"] = ratio  # 将尺寸调整比例 存储在 img_info 字典中 “ratio”键中
        # 将预处理后的图像数据转换为 pytorch 张量，并添加一个额外维度表示批处理大小？？？ 额外维度，批处理大小是多少 0？  并将张量移动到指定设备上
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        # 如果使用16位 浮点数 进行推理（self.fp16 位True），将图像张量转换为半精度 浮点数（FP16）格式

        # print("特征图大小12 up:", img.size())  #torch.Size([1, 3, 800, 1440])
        if self.fp16:
            img = img.half()  # to FP16
        # 创建一个上下文环境，禁用梯度计算，确保在推理过程中不会进行参数更新
        with torch.no_grad():
            timer.tic() # 启动计时器，记录推理时间的起点
            outputs = self.model(img) # 将预处理后的图像输入模型进行推理，获取模型的输出
            # 检查是否提供 解码器对象
            if self.decoder is not None:
                # 如果提供了解码器对象，使用解码器对模型的输出进行解码，并将解码后的结果赋值给 outputs
                outputs = self.decoder(outputs, dtype=outputs.type())
                # 对模型输出进行后处理，包括阈值过滤 和非极大值抑制(NMS),以获取最终的预测结果。 进入下面函数查看内部代码
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info # 返回预测结果和图像信息


def image_demo(predictor, vis_folder, current_time, args):
    # 检查 args.path 是否为一个目录，如果是，则调用get_image_list() 函数获取目录中的图像文件列表； 否则 将 args.path 添加到 files 列表中
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort() # 对files 列表中的文件进行排序，确保按照字母顺序处理图像
    # 创建一个 名为 tracker 的 BOTSORT 对象，该对象进行多目标跟踪，并根据传入参数 args 和 fram_rate 进行初始化
    tracker = ROTSORT(args, frame_rate=args.fps)  # key5

    timer = Timer() # 创建一个计时器对象timer ，用于计算每个图像处理的时间
    results = [] # 创建一个空列表results ，用于存储处理后的跟踪结果

    # 使用 enumerate 函数 遍历 files 列表中的图像文件，并使用frame_id 记录当前帧的编号，img_path存储当前帧的图像文件路径。编号从1开始
    for frame_id, img_path in enumerate(files, 1):

        # Detect objects # 使用predictor 对象对当前帧的图像进行推断，返回检测结果 outputs和 图像信息 img_info ， timer 用于计算推断时间
        outputs, img_info = predictor.inference(img_path, timer)
        # 计算缩放比例 scale ，用于将检测结果中的边界框坐标缩放到原始图像尺寸上
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        detections = []  # 创建一个空表 detections 用于 存储处理后的检测结果
        # 检查检测结果是否不为空
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy() # 将检测结果转换为 Numpy数组 ，并将其移动到 cpu 内存
            detections = outputs[:, :7] # 从检测结果中提取前7列，既边界框的 坐标 和 置信度信息
            detections[:, :4] /= scale # 将边界框的坐标除以缩放比例，还原到原始图像尺寸上  ，想法缩放比例可以根据学习？适当缩放，增加跟踪效果？

            # Run tracker # 使用跟踪器对象tracker 对当前帧的检测结果和原始图像进行跟踪更新，返回在线跟踪结果
            online_targets = tracker.update(detections, img_info['raw_img'])

            online_tlwhs = []  # 创建空列表 用于存储目标的位置
            online_ids = []  # 存储目标的ID
            online_scores = []  # 存储目标的 置信度信息 ？？
            # 遍历在线目标列表
            for t in online_targets:
                tlwh = t.tlwh # 获取目标的位置信息，既左上角坐标和宽高
                tid = t.track_id  # 获取目标的唯一跟踪ID
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh   # 计算目标的宽高比是否大于 args.aspect_ratio_thresh,判断目标是否为垂直方向的目标？
                # 检查目标目标的面积是否大于 args.min_box_area 并且不是垂直方向的目标
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    # 将满足条件的的 目标的位置 、ID 和置信度信息添加到对应的列表中
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results  # 将满足条件的目标信息，以特定的格式添加到 results 列表中，用于保存跟踪结果
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()  # 停止计时器，记录当前帧的处理时间
            # plot_tracking 该函数根据，跟踪结果绘制跟踪框和标识，并返回绘制后的图像
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            # 如果检测结果为空执行下面语句
            timer.toc()
            online_im = img_info['raw_img'] # 将原始图像赋值给 online_im  既不进行跟踪绘制
            # online_im = plot_tracking(
            #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            # )


        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        # 检查是否需要保存结果
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time) # 根据当前时间生成时间戳
            save_folder = osp.join(vis_folder, timestamp)  # 生层保存结果的文件夹路径，并将其命名为 timestamp
            os.makedirs(save_folder, exist_ok=True)  # 创建保存结果的文件夹，如果文件夹已经存在，则不会抛出错误
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im) # 将处理后的图像online_im保存在指定的文件夹中，文件名与原始图像文件名相同

        if frame_id % 20 == 0: # 每处理20帧进行一次判断
            # 使用日志记录当前帧的处理信息，包括帧编号和平均处理帧率
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0) # 等待用户键盘输入，当输入0 或其它键时，显示下一帧图像 ，按下面键时，退出显示
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    # 检查是否需要保存结果
    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt") # 生成保存结果的文本文件路径，文件名与时间戳相同
        # 以写的模式打开文本文件
        with open(res_file, 'w') as f:
            # 将结果列表中的内容按行写入文本文件
            f.writelines(results)
        logger.info(f"save results to {res_file}") # 日志记录保存结果的文本文件路径



# 输出检测目标框，到图片帧上
def draw_detections_on_image(detections, img):
    for detection in detections:
        x1, y1, x2, y2, score1, score2, class_id = detection
        # 绘制矩形框 在当前帧上
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 绘制标签和得分
        # label = f'ID: {int(class_id)}, Score1: {score1:.2f}, Score2: {score2:.2f}'   # label 标签，内容会显示到 当前帧图像上
        label = f'ID: {int(class_id)}'   # 仅绘制id
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # 输出框框信息
        print(
            f'Detected ID: {int(class_id)} with scores {score1:.2f}, {score2:.2f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]')


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")

    save_path = "/tools/output_frames/MOT20/MOT20-02.mp4"
    fps = 30  # 帧率

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    tracker = ROTSORT(args, frame_rate=args.fps)
    # tracker1 = BoTSORT1(args, frame_rate=args.fps)   # deepsort 的KF

    timer = Timer()
    frame_id = 0
    # frame_id = 1  # 跟踪结果 开始为第一帧，这个0和1有什么区别？？ 为什么不能测试txt？？
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            # Detect objects
            outputs, img_info = predictor.inference(frame, timer)
            scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= scale

                # draw_detections_on_image(detections, img_info["raw_img"])  # 调用显示框框
                # Run tracker
                online_targets = tracker.update(detections, img_info["raw_img"])
                # online1_targets = tracker1.update(detections, img_info["raw_img"])



                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                # 副本添加，显示另一个卡尔曼估计结果   deepsort的KF
                # online1_tlwhs = []
                # online1_ids = []
                # for t in online1_targets:
                #     tlwh = t.tlwh
                #     tid = t.track_id
                #     vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                #     if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                #         online1_tlwhs.append(tlwh)
                #         online1_ids.append(tid)


                timer.toc()
                # plot_tracking 该函数可以将 跟踪框，id，轨迹，等信息绘制在原图像，用于观察显示
                # online_im = plot_tracking(
                #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                # )

                # 只显示 框框
                # online_im = plot_trackingk(
                #     img_info['raw_img'], online_tlwhs, online_ids, online1_tlwhs, online1_ids
                # )

                # 不同KF 显示不同框框表示  deepsort的KF 是online1_tlwhs
                # online_im = plot_trackingkTD(
                #     img_info['raw_img'], online_tlwhs, online_ids, online1_tlwhs, online1_ids
                # )

                # 只显示 框框 和 id
                # online_im = plot_tracking(
                #     img_info['raw_img'],  online_tlwhs, online_ids
                # )

                # 显示检测框、分数 .检测框颜色相同
                # online_im = plot_tracking1235(
                #     img_info['raw_img'], online_tlwhs,online_scores,frame_id=frame_id
                # )
                #
                # 显示检测框、分数，检测框颜色不同
                # online_im = plot_tracking121(
                #     img_info['raw_img'], online_tlwhs,online_scores,frame_id=frame_id
                # )
                online_im = plot_tracking121(
                    img_info['raw_img'], online_tlwhs, online_scores, frame_id=frame_id, ids=online_ids
                )






                # cv2.imshow('Tracking MOT20-02', online_im)
                # # 等待键盘输入，键盘输入0 时关闭当前帧，显示下一帧跟踪结果
                # cv2.waitKey(0)
                # # # 关闭图像窗口
                # cv2.destroyAllWindows()
                # # 将显示的图像 保存到指定文件夹
                # cv2.imwrite(f'/home/www/code/MOT/BoT-SORT/BoT-SORT-main/tools/output_frames/MOT20-02/frame_{frame_id}.jpg', online_im)

                cv2.imshow("Tracking MOT-02", online_im)

                # 保存当前帧到视频文件
                vid_writer.write(online_im)

                # 等待用户输入控制播放
                key = cv2.waitKey(0)
                if key == 27:  # 按下 ESC 键退出
                    break

                # print("online_im:", online_im,"online_ids:",online_ids,"online_score:",online_scores,
                #       "online_tlwhs:",online_tlwhs,"online_targets:",online_targets) # 控制台输出，看到的框框id，因为我们看到的是 界面是上面代码绘制上去的，因此在此处可以输出，绘制id ，这个与真实的next_id 要分清
                # 控制台输出，看到的框框id相关信息
                # 假设所有列表长度不同，为了保持输出的对齐性，需要找出最长的列表，不足的话就用None填充。
                # max_length = max(len(online_ids), len(online_scores), len(online_tlwhs),
                #                  len(online_targets))
                #
                # for i in range(max_length):
                #     # im = online_im[i] if i < len(online_im) else None
                #     ids = online_ids[i] if i < len(online_ids) else None
                #     scores = online_scores[i] if i < len(online_scores) else None
                #     tlwhs = online_tlwhs[i] if i < len(online_tlwhs) else None
                #w
                #
                #     print(f"online_ids: {ids}, online_scores: {scores}, online_tlwhs: {tlwhs}")

                # 显示图片
                # cv2.imshow('Detections', img_info["raw_img"])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # 将检测信息输出
                # with open('output.txt', 'w') as f:
                #     max_length = max(len(online_ids), len(online_scores), len(online_tlwhs))
                #
                #     for i in range(max_length):
                #
                #         ids = online_ids[i] if i < len(online_ids) else None
                #         scores = online_scores[i] if i < len(online_scores) else None
                #         tlwhs = online_tlwhs[i] if i < len(online_tlwhs) else None
                #
                #
                #         line = f" online_ids: {ids}, online_scores: {scores}, online_tlwhs: {tlwhs}\n"
                #         f.write(line)


                # # 显示图像 （显示绘制的图像）
            else:
                timer.toc()
                online_im = img_info['raw_img']
            # online_im = img_info['raw_img']    # 只显示当前图片
            # if args.save_result:
            #     vid_writer.write(online_im)

            # cv2.imshow("named",online_im) # 添加显示 ，该显示，当按下关键时，显示退出
            # # 下面一行代码是控制显示速度，当 waitKey 内数值为 1000时 每秒显示1帧跟踪结果
            # ch = cv2.waitKey(1000)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break


            # cv2.imshow('Tracking ResultsTD', online_im)
            # # 等待键盘输入，键盘输入0 时关闭当前帧，显示下一帧跟踪结果
            # cv2.waitKey(0)
            # # # 关闭图像窗口
            # cv2.destroyAllWindows()
            # # 将显示的图像 保存到指定文件夹
            # cv2.imwrite(f'output_frames/frame_{frame_id}.jpg', online_im)

        else:
            break
        frame_id += 1




    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis1")
        os.makedirs(vis_folder, exist_ok=True)

    # 有GPU 选择GPU 没有选择 CPU
    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))   # 输出相应的参数信息

    # 如果 args.conf 不为 None 执行if中的语句
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)  # 实验配置中用于指定测试图像尺寸参数

    model = exp.get_model().to(args.device) # 调用实验配置对象exp 的 get_model() 方法，获取模型对象，并将其移动到指定的设备中
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))  # 使用日志记录器 logger 输出模型的摘要信息
    model.eval() # 将模型设置为评估模式。该代码通常用于，确保在测试阶段使用模型时，模型不会应用任何训练相关的操作，如批归一化的统计信息更新和激活

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint") # 使用日志记录器 logger 输出信息，表示正在加载 模型
        ckpt = torch.load(ckpt_file, map_location="cpu") # 使用pytorch的 torch.load() 函数加载模型文件
        # load the model state dict
        model.load_state_dict(ckpt["model"]) # 从加载的检查点中提取模型的状态字典，并通过调用模型对象的 load_state_dict方法加载模型参数
        # 上面一行代码，用于将检查点中保存的模型参数恢复到当前的模型中
        logger.info("loaded checkpoint done.")  # 输出加载检查点文件完成 ，既模型加载完毕

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
# 创建 一个 Predictor 对象，用于进行目标跟踪的预测。model 是已加载的模型对象，exp 是实验配置对象，trt_file 是TensorRT 文件路径，用于加速推理
# decoder 解码器对象，args.device 是指定的设备，args.fp16 是一个布尔值，表示是否使用16位 浮点数进行推理
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)    # key1
    current_time = time.localtime()  # 获取当前本地时间，存储在current_time 变量中
    #下面代码是根据命令行参数args.demo 的值，选择执行图像目标跟踪， 还是 视频或摄像头目标跟踪，并调用相应的函数进行操作，最终将结果保存在指定的文件夹中
    if args.demo == "image" or args.demo == "images":   # 表示图像目标跟踪演示      # key2
        image_demo(predictor, vis_folder, current_time, args) # 调用 image_demo 函数 传递predictor 对象，vis_folder 文件夹路径，current_time 变量, args参数
    elif args.demo == "video" or args.demo == "webcam":    # 表示视频或摄像头目标跟踪的演示
        imageflow_demo(predictor, vis_folder, current_time, args)    # key3
    else:
        raise ValueError("Error: Unknown source: " + args.demo)  # 表示出现未知的来源（args.demo 的值）改代码用于处理未知来源的错误情况


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
