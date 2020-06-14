import cv2
import numpy as np
import os
from tqdm import tqdm
class Video2ImageSequence:
    def __init__(self, interval = None):
        self.interval = interval

    def __call__(self, video_path, image_save_path, start = 0, end = None):
        v_cap = cv2.VideoCapture(video)
        fps = v_cap.get(cv2.CAP_PROP_FPS)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = v_len if end is None else end
        if self.interval is None:
            sampler = np.arange(start, end, int(fps))
        else:
            ampler = np.arange(start, end, interval)

        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)

        for i in tqdm(range(v_len)):
            success = v_cap.grab()
            if i in sampler:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                cv2.imwrite(f'{image_save_path}/{i}.png', frame)
        v_cap.release()
def Top2Dict(path):
    all_dets = np.loadtxt(path ,delimiter=',')
    frame_num = all_dets[:, 1].astype(int)
    rst = {}
    for i in sorted(set(frame_num)):
        rst[i] = all_dets[frame_num == i][:,-4:]
    return rst