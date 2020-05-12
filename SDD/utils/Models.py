'''
step0 install gluoncv
pip install --upgrade mxnet gluoncv
'''
import mxnet as mx
import numpy as np
import cv2
import os
import gluoncv
import math

from matplotlib import pyplot as plt
from mxnet import nd
from copy import deepcopy
from tqdm import tqdm
from gluoncv import model_zoo, data, utils

get_model = {
    'yolo_v3': lambda device: model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx = device),
    'ssd': lambda device: model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx = device),
    'faster_rcnn': lambda device: model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx = device),
    'center_net': lambda device: model_zoo.get_model('center_net_resnet101_v1b_voc', pretrained=True, ctx = device)
}

image_transform = {
    'yolo_v3': lambda image: data.transforms.presets.yolo.transform_test(image),
    'ssd': lambda image: data.transforms.presets.ssd.transform_test(image, short=512),
    'faster_rcnn': lambda image: data.transforms.presets.rcnn.transform_test(image),
    'center_net': lambda image: data.transforms.presets.center_net.transform_test(image, short = 512)
}
class Bbox_detector:
    def __init__(self,selected_model, transformer = None, device = mx.gpu()):
        self.device = device
        self.transformer = transformer
        self.model = selected_model
        self.net = get_model[selected_model](self.device)

    def __call__(self, image, threshold = 0.7, display = False):
        '''get bbox for input image'''
        image = nd.array(image)
        x, orig_img = image_transform[self.model](image)
        self.shape = orig_img.shape[:2]
        self.benchmark = max(orig_img.shape[:2])
        x = x.copyto(self.device)
        box_ids, scores, bboxes = self.net(x)
        bboxes = bboxes * (image.shape[0]/orig_img.shape[0])
        person_index = []

        if self.model == 'center_net':
            box_ids = box_ids.reshape((1, -1, 1))
            scores = scores.reshape((1, -1, 1))

        #check person class
        for i in range(box_ids.shape[1]):
            if box_ids[0][i][0] == 14 and scores[0][i][0] > threshold:
                person_index.append(i)

        if len(person_index) == 0:
            image = image.astype('uint8')
            return image.asnumpy(), [], [], [], [], []
        #select bbox of person
        #p1:bbox id of person
        #p2:confidence score
        #p3:bbox location
        p1,p2,p3 = box_ids[0][[person_index],:],scores[0][[person_index],:],bboxes[0][[person_index],:]

        #calaulate bbox coordinate
        bbox_center = self.bbox_center(p3)
        img_with_bbox = utils.viz.cv_plot_bbox(image.astype('uint8'), p3[0], p2[0], p1[0], colors={14: (0,255,0)},class_names = self.net.classes, linewidth=3)
        if self.transformer:
            result_img, edges = self.bbox_distance(img_with_bbox, bbox_center)
        else:
            bbox_height = self.bbox_height(p3)
            result_img, edges = self.bbox_distance(img_with_bbox, bbox_center, bbox_height)

        if display:
          plt.imshow(result_img)
          plt.show()
        return result_img, p1, p2, p3, bbox_center, edges

    def show(self, img, p1, p2, p3, bbox_center, edges, resize = None):
        if resize is not None:
            img = mx.image.imresize(nd.array(img).astype('uint8'), self.shape[1], self.shape[0])
        else:
            img = nd.array(img).astype('uint8')

        if len(p1) == 0: return img.asnumpy()

        img_with_bbox = utils.viz.cv_plot_bbox(img, p3[0], p2[0], p1[0], colors={14: (0,255,0)},class_names = self.net.classes, linewidth=1)
        for coor in range(len(bbox_center)):
            cv2.circle(img_with_bbox, (int(bbox_center[coor][0]), int(bbox_center[coor][1])), 5 ,(0, 0, 255), -1)
        
        for edge in edges:
            cv2.line(img_with_bbox, *edge, (255, 0, 0), 2)

        return img_with_bbox

    def bbox_center(self, bbox_location):
        '''calculate center coordinate for each bbox'''
        center_x = (bbox_location[:, :, 0]+bbox_location[:, :, 2])/2
        center_x = center_x.reshape((-1, 1))
        center_y = bbox_location[:, :, 3].reshape((-1, 1))
        return nd.concatenate([center_x, center_y], axis = 1).asnumpy()
    
    def bbox_height(self, bbox_location):
        return (bbox_location[:,:,3] - bbox_location[:, :, 1]).reshape((-1,)).asnumpy()

    def bbox_distance(self, img, bbox_center_coord, bbox_height_coord = None, max_detect = 4, safe= 1.5):
        '''
        calculate distance between each bbox, 
        if distance < safe, draw a red line
        '''
        #draw the center
        safe = safe**2
        max_detect = max_detect**2
        for coor in range(len(bbox_center_coord)):
            cv2.circle(img,(int(bbox_center_coord[coor][0]),int(bbox_center_coord[coor][1])),5,(0, 0, 255),-1)

        
        # print(bird_eye_view)
        # self.transformer.imshow(img)
        edges = set()

        for i in range(0, len(bbox_center_coord)):
            for j in range(i+1, len(bbox_center_coord)):
                # print('TRansife ms ',transform)
                if self.transformer:
                    bird_eye_view = self.transformer(deepcopy(bbox_center_coord))
                    dist = self.transformer.distance(bird_eye_view[i], bird_eye_view [j])
                    if dist < safe:
                        edges.add(((bbox_center_coord[i][0], bbox_center_coord[i][1]), (bbox_center_coord[j][0], bbox_center_coord[j][1])))
                        cv2.line(img,(bbox_center_coord[i][0],bbox_center_coord[i][1]),(bbox_center_coord[j][0],bbox_center_coord[j][1]),(255, 0, 0), 2)
                else:
                    dist = (bbox_center_coord[i][0] - bbox_center_coord[j][0])**2 + (bbox_center_coord[i][1] - bbox_center_coord[j][1])**2
                    # print('height vs dist',height.asscalar(),dist)
                    if dist <= ((bbox_height_coord[i] + bbox_height_coord[j])/2)**2:
                        edges.add(((bbox_center_coord[i][0], bbox_center_coord[i][1]), (bbox_center_coord[j][0], bbox_center_coord[j][1])))
                        cv2.line(img,(bbox_center_coord[i][0],bbox_center_coord[i][1]),(bbox_center_coord[j][0],bbox_center_coord[j][1]),(255, 0, 0), 2)   
        return img, edges


#result <class 'numpy.ndarray'>
#no bbox <class 'numpy.ndarray'>