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

class Bird_eye_view_Transformer:
    def __init__(self, keypoints, keypoints_birds_eye_view, actual_length, actual_width, multi_pts = False):
        '''
        keypoints input order 
        0   1

        2   3
        '''
        if not multi_pts:
          self.keypoint = np.float32(keypoints)
          self.keypoints_birds_eye_view = np.float32(keypoints_birds_eye_view) 
          self.M = cv2.getPerspectiveTransform(self.keypoint, self.keypoints_birds_eye_view)
        else:
          self.keypoint = np.float32(self.generate_grid(keypoints))
          self.keypoints_birds_eye_view = np.float32(self.generate_grid(keypoints_birds_eye_view))
          self.M, self.mask = cv2.findHomography(self.keypoint, self.keypoints_birds_eye_view, cv2.RANSAC)
        # print(keypoints_birds_eye_view[0],keypoints_birds_eye_view[-1])
        self.width_ratio = actual_width/(keypoints_birds_eye_view[-1][0] - keypoints_birds_eye_view[0][0])
        self.length_ratio = actual_length/(keypoints_birds_eye_view[-1][1] - keypoints_birds_eye_view[0][1])
        print('camera: real-world = (length,width ratio) ',self.length_ratio, self.width_ratio)

    def imshow(self, img):
        dst_img = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
        plt.imshow(dst_img)
        plt.show()

    def generate_grid(self, keypoints, nw = 5, nh = 5):
      height_left = abs(keypoints[2][1]-keypoints[0][1]) // nh
      height_right = abs(keypoints[3][1]-keypoints[1][1]) // nh

      width_left = abs(keypoints[2][0]-keypoints[0][0]) // nw 
      width_right = abs(keypoints[3][0]-keypoints[1][0]) // nw 
      rst = []
      for j in range(nh+1):
        row_start = (keypoints[0][0] - j * width_left, keypoints[0][1] + j * height_left)
        row_end = (keypoints[1][0] - j * width_right, keypoints[1][1] + j * height_right)
        width_top = abs(row_start[0]- row_end[0])//nw
        width_bottom = abs(row_start[1]- row_end[1])//nw
        
        for i in range(nw+1):
          new_pt_x = row_start[0]+i*width_top
          new_pt_y = row_start[1]+i*width_bottom
          # print('pt are ',(new_pt_x,new_pt_y))
          rst.append((new_pt_x,new_pt_y))

      return rst

    def __call__(self, points):
        h = points.shape[0]
        points = np.concatenate([points, np.ones((h, 1))], axis = 1)
        temp = self.M.dot(points.T)
        return (temp[:2]/temp[2]).T
    
    def distance(self, p0, p1):
        return ((p0[0] - p1[0])*self.width_ratio)**2 \
        + ((p0[1] - p1[1])*self.length_ratio)**2