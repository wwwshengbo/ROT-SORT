
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

# 线性插值
def LinearInterpolation(input_, interval):
    # 按ID和帧排序
    input_ = input_[np.lexsort((input_[:, 0], input_[:, 1]))]
    ids = np.unique(input_[:, 1])
    output_ = []
    for id_ in ids:
        track = input_[input_[:, 1] == id_]
        frames = track[:, 0].astype(int)
        positions = track[:, 2:6]
        # 创建完整的帧集
        full_frames = np.arange(frames.min(), frames.max() + 1)
        full_positions = np.zeros((len(full_frames), positions.shape[1]))
        # 执行线性插值
        for i, frame in enumerate(full_frames):
            if frame in frames:
                full_positions[i] = positions[frames == frame]
            else:
                prev_idx = np.max(np.where(frames < frame)[0])
                next_idx = np.min(np.where(frames > frame)[0])
                if np.isnan(prev_idx) or np.isnan(next_idx):
                    continue
                prev_frame, next_frame = frames[prev_idx], frames[next_idx]
                prev_pos, next_pos = positions[prev_idx], positions[next_idx]
                factor = (frame - prev_frame) / (next_frame - prev_frame)
                full_positions[i] = prev_pos + factor * (next_pos - prev_pos)
        # 添加到输出列表
        for i, frame in enumerate(full_frames):
            output_.append([frame, id_, *full_positions[i], 1, -1, -1, -1])
    return np.array(output_)

# 高斯平滑
def GaussianSmooth(input_, tau):
    ids = np.unique(input_[:, 1])
    output_ = []

    for id_ in ids:
        track = input_[input_[:, 1] == id_]
        t = track[:, 0].reshape(-1, 1)
        x = track[:, 2].reshape(-1, 1)
        y = track[:, 3].reshape(-1, 1)
        w = track[:, 4].reshape(-1, 1)
        h = track[:, 5].reshape(-1, 1)
        len_scale = np.clip(tau * np.log(tau ** 3 / len(t)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        for feature, values in zip([x, y, w, h], [x, y, w, h]):
            gpr.fit(t, feature)
            smoothed_values = gpr.predict(t)
            feature[:] = smoothed_values
        output_.extend([
            [t[i, 0], id_, x[i, 0], y[i, 0], w[i, 0], h[i, 0], 1, -1, -1, -1]
            for i in range(len(t))
        ])
    return np.array(output_)

# GSI
def GSInterpolation1(path_in, path_out, interval, tau):
    input_ = np.loadtxt(path_in, delimiter=',')
    li = LinearInterpolation(input_, interval)
    gsi = GaussianSmooth(li, tau)
    np.savetxt(path_out, gsi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d')
# 示例使用方法
# GSInterpolation('path_to_input_folder', 'path_to_output_folder', interval=10, tau=1.0)

if __name__ == "__main__":

    in_path = "/home/www/code/MOT/BoT-SORT/BoT-SORT-main/YOLOX_outputs/yolox_x_mix_det/track_results_MOT17_train_T3/MOT17-02-DPM.txt"
    out_path = "/home/www/code/MOT/BoT-SORT/BoT-SORT-main/YOLOX_outputs/yolox_x_mix_det/track_results_MOT17_train_T3/in_MOT17-02-DPM.txt"
    # 示例使用方法
    GSInterpolation1(in_path, out_path, interval=10, tau=1.0)
