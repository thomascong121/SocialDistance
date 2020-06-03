import mxnet as mx
import numpy as np
import cv2
from skimage import io
import os
import gluoncv
import csv
import re 

from matplotlib import pyplot as plt
from mxnet import nd
from copy import deepcopy
from tqdm import tqdm
from gluoncv import model_zoo, data, utils
def accur_metric(TP, TN, FP, FN):
    precision = 1 if TP + FP == 0 else TP / (TP + FP)
    recall = 1 if TP + FN == 0 else TP / (TP + FN)
    return precision, recall
class VideoDetector:  
    def __init__(self, model, threshold = 0.7, save_path = './detections', batch_size = 60, interval = None):
        self.detector = model
        self.save_path = save_path
        self.interval = interval
        self.threshold = threshold
        self.batch_size = batch_size

    def __call__(self, filename, groundTruth = None, metric = 'mean', scale = 1):
        all_dets = distance_gt = None
        # if groundTruth:
        #     all_dets = np.loadtxt(groundTruth, delimiter = ',')
        #     distance_gt = list(map(int, list(set(all_dets[:,1]))))
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (v_cap.get(cv2.CAP_PROP_FRAME_WIDTH), v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = v_cap.get(cv2.CAP_PROP_FPS)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        out = cv2.VideoWriter(f'{self.save_path}/{filename.split("/")[-1]}', fourcc, fps,\
                            (int(frame_size[0]), int(frame_size[1])))   
        
        # if self.interval is None:
        #     sample = np.arange(0, v_len)
        # else:
        #     sample = np.arange(0, v_len, self.interval)
        frame = p1 = p2 = p3 = bbox_center = edges = None
        TP = FP = TN = FN = extra = 0
        precision_per_frame = np.array([])
        recall_per_frame = np.array([])
        # print('add_det',distance_gt)
        for i in tqdm(range(v_len)):#200
            success = v_cap.grab()
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # if i in sample:
            frame, _TP, _FP, _TN, _FN, _extra = self.detector(frame, groundTruth, metric = metric, scale = scale, frame_number = i, threshold = self.threshold)
            # else:
            #     frame = self.detector.show(frame, p1, p2, p3, bbox_center, edges)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            TP += _TP
            FP += _FP
            TN += _TN
            FN += _FN
            extra += _extra
            precision_frame, recall_frame = accur_metric(TP, TN, FP, FN)
            precision_per_frame = np.hstack((precision_per_frame, [precision_frame]))
            recall_per_frame = np.hstack((recall_per_frame, [recall_frame]))
        print(f'Image saved at {self.save_path}/{filename.split("/")[-1]}')
        v_cap.release()
        if groundTruth != None:
            # precision_video, recall_video = metric(TP, TN, FP, FN)
        # print(' ===== Video Detection Done, Precision is {0} and recall is {1}: '.format(precision_video, recall_video))
            print('\n Mean Average Precision and Recall is', np.mean(precision_per_frame), np.mean(recall_per_frame))
        print(f'Video saved at {self.save_path}/{filename.split("/")[-1]}')
        return out, TP, FP, TN, FN, extra
class ImageDetector:  
    def __init__(self, model, threshold = 0.7, save_path = './detections', batch_size = 60):
        self.detector = model
        self.save_path = save_path
        self.threshold = threshold
        self.batch_size = batch_size

    def __call__(self, imgpath, groundTruth=None, metric = 'mean', scale = 1):
        all_dets = distance_gt = None
        if groundTruth:
            all_dets = np.loadtxt(groundTruth, delimiter = ',')
            distance_gt = list(map(int, list(set(all_dets[:,1]))))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        frames = os.listdir(imgpath)
        frame = p1 = p2 = p3 = bbox_center = edges = None
        # diff = np.array([])
        TP = FP = TN = FN = extra = 0
        precision_per_frame = np.array([])
        recall_per_frame = np.array([])
        for i in tqdm(range(len(frames))):
            try:
                fnumbers = frames[i].split('.')[0]
                fnumbers = int(re.findall(r'[0-9]+', fnumbers)[0])
            except:
                fnumbers = i
            
            if not imgpath + '/' + frames[i]:
                continue
            # print('frnumber is ',imgpath + '/' + frames[i])
            try:
              frame = cv2.imread(imgpath + '/' + frames[i])
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
              continue
            # print('frnumber is ',fnumbers, sorted(distance_gt))
            frame, _TP, _FP, _TN, _FN, _extra  = self.detector(frame, groundTruth, metric = metric, scale = scale, frame_number = fnumbers, threshold = self.threshold)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.save_path + '/saved_'+frames[i], frame)
            # print('write ',self.save_path + '/saved_'+frames[i])

            TP += _TP
            FP += _FP
            TN += _TN
            FN += _FN
            extra += _extra
            precision_frame, recall_frame = accur_metric(TP, TN, FP, FN)
            precision_per_frame = np.hstack((precision_per_frame, [precision_frame]))
            recall_per_frame = np.hstack((recall_per_frame, [recall_frame]))

        if groundTruth != None:
            # precision_video, recall_video = metric(TP, TN, FP, FN)
        # print('\n ===== Video Detection Done, Precision is {0} and recall is {1}: '.format(precision_video, recall_video))
            print('Mean Average Precision and Recall is', np.mean(precision_per_frame), np.mean(recall_per_frame))
       
        print(f'Video saved at {self.save_path}/')
        return None, TP, FP, TN, FN, extra