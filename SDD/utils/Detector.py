import mxnet as mx
import numpy as np
import cv2
import os
import gluoncv

from matplotlib import pyplot as plt
from mxnet import nd
from copy import deepcopy
from tqdm import tqdm
from gluoncv import model_zoo, data, utils


class Detector:  
  def __init__(self, model, save_path = './detections', batch_size = 60, interval = None):
    self.detector = model
    self.save_path = save_path
    self.interval = interval
    self.batch_size = batch_size

  def __call__(self, filename):
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (v_cap.get(cv2.CAP_PROP_FRAME_WIDTH), v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    if not os.path.exists(self.save_path):
        os.mkdir(self.save_path)
    print(f'{self.save_path}/{filename.split("/")[-1]}')
    out = cv2.VideoWriter(f'{self.save_path}/{filename.split("/")[-1]}', fourcc, fps,\
                          (int(frame_size[0]), int(frame_size[1])))   
    
    if self.interval is None:
      sample = np.arange(0, v_len)
    else:
      sample = np.arange(0, v_len, self.interval)
    frame = p1 = p2 = p3 = bbox_center =None
    for i in tqdm(range(v_len)):
      success = v_cap.grab()
      
      success, frame = v_cap.retrieve()
      if not success:
        continue
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      if i in sample:
        frame, p1, p2, p3, bbox_center = self.detector(frame)
      else:
        frame = self.detector.show(frame, p1, p2, p3, bbox_center)
      # plt.imshow(frame)
      # plt.show()
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      out.write(frame)

    v_cap.release()
    return out