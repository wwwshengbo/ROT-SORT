import cv2
import os
from natsort import natsorted  # For natural sorting of file names


def images_to_video(image_folder, output_video, fps=30):
    # 获取图像文件列表并排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = natsorted(images)  # 使用自然排序以确保按顺序读取文件

    if not images:
        print("No images found in the directory.")
        return

    # 读取第一张图片来确定视频的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        print("Could not read the first image.")
        return

    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式

    # 初始化视频写入器
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 遍历图片文件并写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Could not read image {image_name}. Skipping...")
            continue

        video.write(frame)  # 写入帧

    # 释放视频写入器
    video.release()
    print(f"Video has been saved as {output_video}")


# 设置图像文件夹路径和输出视频文件路径
# image_folder = "/path/to/your/image/folder"
# output_video = "output_video.mp4"
image_folder = "/home/www/Dataset/MOT_Dataset/MOT20/MOT20/train/MOT20-02/img1"
output_video = "/home/www/Dataset/MOT_Dataset/MOT20/MOT20/train/MOT20-02/MOT20-02.mp4"



# 调用函数生成视频
images_to_video(image_folder, output_video)
