import gluoncv
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy
from collections import defaultdict
import cv2
'''
step0 install gluoncv
pip install --upgrade mxnet gluoncv
'''
class Model_Zoo:
    def __init__(self,selected_model):
        print('=======using pretrained {0}======'.format(selected_model))
        self.smodel = selected_model
    def detect(self,image,display=False):
        '''get bbox for input image'''
        net = model_zoo.get_model(self.smodel, pretrained=True)
        x, orig_img = data.transforms.presets.rcnn.load_test(image)
        box_ids, scores, bboxes = net(x)
        #possible classes:
        #print(net.classes)
        #person = 14
        person_index = []
        for i in range(box_ids.shape[1]):
            if box_ids[0][i][0] == 14 and scores[0][i][0] > 0.5:
                person_index.append(i)
        #select bbox of person
        #p1:bbox id of person
        #p2:confidence score
        #p3:bbox location
        print('======{0} bbox of persons are detected===='.format(len(person_index)))
        p1,p2,p3 = box_ids[0][[person_index],:],scores[0][[person_index],:],bboxes[0][[person_index],:]
        #calaulate bbox coordinate
        bbox_center = self.bbox_center(p3)
        #img with bbox 
        img_with_bbox = utils.viz.cv_plot_bbox(orig_img, p3[0], p2[0], p1[0], colors={14: (255, 0, 0)},class_names=net.classes,linewidth=1)
        result_img = self.bbox_distance(bbox_center,img_with_bbox)
        if display:
            cv2.imshow('img',result_img)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
        return bbox_center
        
    def bbox_center(self,bbox_location):
        '''calculate center coordinate for each bbox'''
        rst = []
        for loc in range(bbox_location[0].shape[0]):
            (xmin, ymin, xmax, ymax) = bbox_location[0][loc]
            center_x = (xmin+xmax)/2
            center_y = (ymin+ymax)/2
            rst.append([center_x.asnumpy()[0],center_y.asnumpy()[0]])
        return rst
    def bbox_distance(self,bbox_coord,img,max_detect=20000,safe=2000):
        '''
        calculate distance between each bbox, 
        if distance < max_detect, draw a green line
        if distance < safe, draw a red line
        '''
        bbox_coord.sort(key=lambda x:x[0]**2+x[1]**2)
        for coor in range(len(bbox_coord)):
            cv2.circle(img,(int(bbox_coord[coor][0]),int(bbox_coord[coor][1])),5,(0,255,0),-1)
            # cv2.putText(img, "bbox {0}".format(coor), (int(rst[coor][0]),int(rst[coor][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        for i in range(0,len(bbox_coord)):
            minm = float('inf')
            index = None
            for j in range(0,len(bbox_coord)):
                if i != j:
                    dist = (bbox_coord[i][0] - bbox_coord[j][0])**2 + (bbox_coord[i][1] - bbox_coord[j][1])**2
                    if  dist < minm:
                        minm = dist
                        index = j
            if minm <= max_detect:
                cv2.line(img,(bbox_coord[i][0],bbox_coord[i][1]),(bbox_coord[index][0],bbox_coord[index][1]),(0,255,0),2)
            if minm <= safe:
                cv2.line(img,(bbox_coord[i][0],bbox_coord[i][1]),(bbox_coord[index][0],bbox_coord[index][1]),(0,0,255),2)
        return img


if __name__=='__main__':
    test_img = '../video_frames/0.png'
    pretrained_models = ['faster_rcnn_resnet50_v1b_voc','ssd_512_resnet50_v1_voc','yolo3_darknet53_voc']
    bbox_center = defaultdict(list)
#     for m in pretrained_models:
#         detect_model = Model_Zoo(m)
#         rst = detect_model.detect(test_img,display=True)
#         bbox_center[m].append(rst)
    detect_model = Model_Zoo(pretrained_models[2])
    rst = detect_model.detect(test_img,display=True)
    bbox_center[pretrained_models[2]].append(rst)

