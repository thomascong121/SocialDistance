'''
step0 install gluoncv
pip install --upgrade mxnet gluoncv
'''
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

class Bbox_detector:
    def __init__(self,selected_model, transformer, device):
        self.device = device
        self.transformer = transformer
        self.net = model_zoo.get_model(selected_model, pretrained=True, ctx = self.device)

    def __call__(self,image,display=False):
        '''get bbox for input image'''
        image = nd.array(image)
        x, orig_img = data.transforms.presets.yolo.transform_test(image)
        self.shape = orig_img.shape[:2]
        self.benchmark = max(orig_img.shape[:2])
        x = x.copyto(self.device)
        box_ids, scores, bboxes = self.net(x)
        bboxes = bboxes * (image.shape[0]/orig_img.shape[0])
        person_index = []

        #check person class
        for i in range(box_ids.shape[1]):
            if box_ids[0][i][0] == 14 and scores[0][i][0] > 0.7:
                person_index.append(i)
        #select bbox of person
        #p1:bbox id of person
        #p2:confidence score
        #p3:bbox location
        p1,p2,p3 = box_ids[0][[person_index],:],scores[0][[person_index],:],bboxes[0][[person_index],:]
        #calaulate bbox coordinate
        bbox_center = self.bbox_center(p3)
        #img with bbox 

        img_with_bbox = utils.viz.cv_plot_bbox(image.astype('uint8'), p3[0], p2[0], p1[0], colors={14: (0,255,0)},class_names = self.net.classes, linewidth=3)
        result_img = self.bbox_distance(bbox_center,img_with_bbox)
        if display:
          plt.imshow(result_img)
          plt.show()
        return result_img, p1, p2, p3, bbox_center

    def show(self, img, p1, p2, p3, bbox_center, resize = None):
        if resize is not None:
            img = mx.image.imresize(nd.array(img).astype('uint8'), self.shape[1], self.shape[0])
        else:
            img = nd.array(img).astype('uint8')
        img_with_bbox = utils.viz.cv_plot_bbox(img, p3[0], p2[0], p1[0], colors={14: (0,255,0)},class_names = self.net.classes, linewidth=1)
        return self.bbox_distance(bbox_center,img_with_bbox)

    def bbox_center(self,bbox_location):
        '''calculate center coordinate for each bbox'''
        rst = None
        for loc in range(bbox_location[0].shape[0]):
            (xmin, ymin, xmax, ymax) = bbox_location[0][loc].copyto(mx.cpu())
            center_x = (xmin+xmax)/2
            center_y = ymax
            if rst is not None:
                rst = nd.concatenate([rst, nd.stack(center_x, center_y, axis = 1)])
            else:
                rst = nd.stack(center_x, center_y, axis = 1)

        return rst.asnumpy()

    def bbox_distance(self,bbox_coord,img, max_detect = 4, safe= 1.5):
        '''
        calculate distance between each bbox, 
        if distance < safe, draw a red line
        '''
        #draw the center
        safe = safe**2
        max_detect = max_detect**2
        for coor in range(len(bbox_coord)):
            cv2.circle(img,(int(bbox_coord[coor][0]),int(bbox_coord[coor][1])),5,(0, 0, 255),-1)

        bird_eye_view = self.transformer(deepcopy(bbox_coord))
        # print(bird_eye_view)
        # self.transformer.imshow(img)

        for i in range(0, len(bbox_coord)):
            for j in range(i+1, len(bbox_coord)):
                dist = self.transformer.distance(bird_eye_view[i], bird_eye_view [j])
                # print(bird_eye_view[i], bird_eye_view [j],dist)
                if dist < safe:
                    cv2.line(img,(bbox_coord[i][0],bbox_coord[i][1]),(bbox_coord[j][0],bbox_coord[j][1]),(255, 0, 0), 2)

        return img