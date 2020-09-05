import mxnet as mx
import numpy as np
import cv2
# from skimage import io
import os
import gluoncv
import csv
import re 

from matplotlib import pyplot as plt
from mxnet import nd
from copy import deepcopy
from tqdm import tqdm
from gluoncv import model_zoo, data, utils
'''
step0 install gluoncv
pip install --upgrade mxnet gluoncv
'''
from scipy import stats
get_model = {
    'yolo_v3': lambda device: model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx = device),
    'yolo_v3_trained': lambda device: model_zoo.get_model('yolo3_darknet53_custom', classes=['person'], pretrained_base=False, ctx = device),
    'ssd': lambda device: model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx = device),
    'ssd_trained': lambda device: model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=['person'], pretrained_base=False, ctx = device),
    'faster_rcnn': lambda device: model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx = device),
    'center_net': lambda device: model_zoo.get_model('center_net_resnet101_v1b_voc', pretrained=True, ctx = device)
}

image_transform = {
    'yolo_v3': lambda image: data.transforms.presets.yolo.transform_test(image),
    'yolo_v3_trained' : lambda image: data.transforms.presets.yolo.transform_test(image),
    'ssd':lambda image: data.transforms.presets.ssd.transform_test(image, short=512),
    'ssd_trained': lambda image: data.transforms.presets.ssd.transform_test(image, short=512),#
    'faster_rcnn': lambda image: data.transforms.presets.rcnn.transform_test(image),
    'center_net': lambda image: data.transforms.presets.center_net.transform_test(image, short = 512)
}

distance_metric = {
    'mean':lambda x,y : (x+y)/2,
    'min':lambda x,y: min(x,y),
    'max':lambda x,y: max(x,y),
    'scale':lambda a,x:a*x
}

class Bbox_detector:
    def __init__(self,selected_model, trained = True, transformer = None, device = mx.gpu()):
        self.device = device
        self.transformer = transformer
        self.class_person = {}
        
        if trained:
          self.param = '../trained_models/yolo_r_best_422.params' if selected_model == 'yolo_v3' else '../trained_models/ssd_best_repl_34.params'
          self.model = selected_model + '_trained'
          self.net = get_model[self.model](self.device)
          self.class_person[0] = 'person'
          self.net.load_parameters(self.param,ctx=self.device)
        else:
          self.model = selected_model
          self.class_person[14] = 'person'
          self.net = get_model[selected_model](self.device)
        print('selected model is ',self.model)

    def __call__(self, image,  bbox_groundtruth = None, metric = 'mean', scale = 1, frame_number = None, threshold = 0.7, display = False):
        '''get bbox for input image'''
        image = nd.array(image)
        x, orig_img = image_transform[self.model](image)
        self.bbox_gt = bbox_groundtruth if bbox_groundtruth != None else None
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
            class_id = box_ids[0][i][0].asscalar()
            if int(class_id) in self.class_person.keys() and scores[0][i][0] > threshold:
                person_index.append(i)

        if len(person_index) == 0:
            # print('no person found  ',frame_number)
            image = image.astype('uint8')
            return image.asnumpy(), 0, 0, 0, 0, 0
        #select bbox of person
        #p1:bbox id of person
        #p2:confidence score
        #p3:bbox location
        p1,p2,p3 = box_ids[0][[person_index],:], scores[0][[person_index],:], bboxes[0][[person_index],:]
        #calaulate bbox coordinate
        TP = FP = TN = FN = extra = 0
        img_with_bbox = utils.viz.cv_plot_bbox(image.astype('uint8'), p3[0], p2[0], p1[0], colors={list(self.class_person.keys())[0]: (0,255,0)},class_names = self.net.classes, linewidth=3)
        if self.transformer:
            result_img, TP, FP, TN, FN, extra,close_bbox = self.bbox_distance(img_with_bbox, p3, metric, scale, frame_number = frame_number)
        else:
            bbox_height = self.bbox_height(p3)
            result_img, TP, FP, TN, FN, extra,close_bbox = self.bbox_distance(img_with_bbox, p3, metric, scale, bbox_height_coord = bbox_height, frame_number = frame_number)

        if display:
          plt.figure(1,figsize=(16,16))
          plt.imshow(result_img)
          plt.show()
        # bbox_center = self.bbox_center(p3)
        return result_img, TP, FP, TN, FN, extra

    def show(self, img, p1, p2, p3, bbox_center, edges, resize = None):
        if resize is not None:
            img = mx.image.imresize(nd.array(img).astype('uint8'), self.shape[1], self.shape[0])
        else:
            img = nd.array(img).astype('uint8')

        if len(p1) == 0: return img.asnumpy()

        img_with_bbox = utils.viz.cv_plot_bbox(img, p3[0], p2[0], p1[0], colors={list(self.class_person.keys())[0]: (0,255,0)},class_names = self.net.classes, linewidth=1)
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

    def _JS_divergence(self, p, q):
        M = np.add(p, q)
        M = np.multiply(M, 0.5)
        return 0.5*stats.entropy(p, M)+0.5*stats.entropy(q, M)

    def IOU(self, p1, p2):#[xleft, yleft, xright, yright]
        xa = max(p1[0], p2[0])
        ya = max(p1[1], p2[1])
        xb = min(p1[2], p2[2])
        yb = min(p1[3], p2[3])

        intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        area1 = abs(p1[2] - p1[0] + 1) * abs(p1[3] - p1[1] + 1)
        area2 = abs(p2[2] - p2[0] + 1) * abs(p2[3] - p2[1] + 1)
        return intersection / (area1 + area2 - intersection)

    def accuracy_calculation(self, frame_number, bbox_coord, close_bbox):
        TP = FP = TN = FN = extra = 0
        if self.bbox_gt == None:
          return TP, FP, TN, FN, extra
        all_dets = np.loadtxt(self.bbox_gt, delimiter = ',')
        distance_gt = list(map(int, list(set(all_dets[:,1]))))
        if frame_number not in distance_gt:
          return TP, FP, TN, FN, extra
        frame_gt = all_dets[all_dets[:,1] == frame_number]
        _p3 = bbox_coord[0]
        #calculate IOU
        for i in range(len(_p3)):
          _iou = 0
          close = 0
          for j in range(len(frame_gt[:, -4:])):
            inter = self.IOU(_p3[i], frame_gt[:, -4:][j])
            if inter > _iou:
              close = j
              _iou = inter
          if i in close_bbox and _iou != 0: #predicted as close and bbox is labelled in ground truth
            if frame_gt[:, 2][close] == 0:
              TP += 1
            if frame_gt[:, 2][close] == 1: 
              FP += 1
          elif _iou != 0: #predicted as safe and bbox is labelled in ground truth
            if frame_gt[:, 2][close] == 1:
              TN += 1
            if frame_gt[:, 2][close] == 0:
              FN += 1
          else: #predicted bbox is not labelled in ground truth
            extra += 1
        
        return TP, FP, TN, FN, extra
    def bbox_distance(self, img, bbox_coord, metric, scale = 1, bbox_height_coord = None, frame_number = None, safe= 1.5):
        '''
        calculate distance between each bbox, 
        if distance < safe, draw a red line
        '''
        #draw the center
        safe = safe**2
        bbox_center_coord = self.bbox_center(bbox_coord)
        for coor in range(len(bbox_center_coord)):
            cv2.circle(img,(int(bbox_center_coord[coor][0]),int(bbox_center_coord[coor][1])),5,(0, 0, 255),-1)
        close_bbox = []
        dist = None
        for i in range(0, len(bbox_center_coord)):
            for j in range(i+1, len(bbox_center_coord)):
                if self.transformer:
                    bird_eye_view = self.transformer(deepcopy(bbox_center_coord))
                    dist = self.transformer.distance(bird_eye_view[i], bird_eye_view [j])
                    if dist < safe:
                        cv2.rectangle(img, (int(bbox_coord[:,i][:,0].asscalar()),int(bbox_coord[:,i][:,1].asscalar())),(int(bbox_coord[:,i][:,2].asscalar()),int(bbox_coord[:,i][:,3].asscalar())),(255, 0, 0), 3)
                        cv2.rectangle(img, (int(bbox_coord[:,j][:,0].asscalar()),int(bbox_coord[:,j][:,1].asscalar())),(int(bbox_coord[:,j][:,2].asscalar()),int(bbox_coord[:,j][:,3].asscalar())),(255, 0, 0), 3)
                        cv2.line(img,(bbox_center_coord[i][0],bbox_center_coord[i][1]),(bbox_center_coord[j][0],bbox_center_coord[j][1]),(255, 0, 0), 3) 
                        close_bbox += [i] if i not in close_bbox else []
                        close_bbox += [j] if j not in close_bbox else []
                else:
                    dist = (bbox_center_coord[i][0] - bbox_center_coord[j][0])**2 + (bbox_center_coord[i][1] - bbox_center_coord[j][1])**2
                    if metric == 'scale':
                        threshold = distance_metric[metric](scale, bbox_height_coord[i])
                    else:
                        threshold = distance_metric[metric](bbox_height_coord[i], bbox_height_coord[j])
                    if dist <= (threshold)**2:# # $#min(1.2 * bbox_height_coord[i], 1.2 * bbox_height_coord[j])##(bbox_height_coord[i] + bbox_height_coord[j])/2#min(bbox_height_coord[i], bbox_height_coord[j])
                        cv2.rectangle(img, (int(bbox_coord[:,i][:,0].asscalar()),int(bbox_coord[:,i][:,1].asscalar())),(int(bbox_coord[:,i][:,2].asscalar()),int(bbox_coord[:,i][:,3].asscalar())),(255, 0, 0), 3)
                        cv2.rectangle(img, (int(bbox_coord[:,j][:,0].asscalar()),int(bbox_coord[:,j][:,1].asscalar())),(int(bbox_coord[:,j][:,2].asscalar()),int(bbox_coord[:,j][:,3].asscalar())),(255, 0, 0), 3)
                        cv2.line(img,(bbox_center_coord[i][0],bbox_center_coord[i][1]),(bbox_center_coord[j][0],bbox_center_coord[j][1]),(255, 0, 0), 3) 
                        close_bbox += [i] if i not in close_bbox else []
                        close_bbox += [j] if j not in close_bbox else []
        TP, FP, TN, FN, extra = self.accuracy_calculation(frame_number, bbox_coord, close_bbox)
        return img, TP, FP, TN, FN, extra, close_bbox