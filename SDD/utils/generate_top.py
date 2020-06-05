import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

class generate_top:
    def label(self, data, annotation, output_file):
        data_list = os.listdir(data)
        gt_json = open(annotation)
        gt_data = json.load(gt_json)
        f = open(output_file, "w")
        for i in data_list:
            frame_number = i.split('.')[0]
            end = i.split('.')[-1]
            try:
            img = cv2.imread(data + '/' + i)
            except:
            continue
            for i in range(len(gt_data[frame_number])):
            cv2.rectangle(img, (gt_data[frame_number][i][0],  gt_data[frame_number][i][1]), (gt_data[frame_number][i][2], gt_data[frame_number][i][3]), (255, 0, 0), 2)
            cv2.putText(img, 'ID:{0}'.format(i), (int(gt_data[frame_number][i][0]), int(gt_data[frame_number][i][1])), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            plt.figure(figsize=(16,16))
            plt.imshow(img)
            plt.show()
            print('frame number: ',frame_number)
            print('total bbox :',len(gt_data[frame_number]))
            close_bbox = input('Enter close index: ')
            close_bbox = list(map(int, close_bbox.split()))
            for i in range(len(gt_data[frame_number])):
            if i in close_bbox:
                f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(i, frame_number, 0, 1, int(gt_data[frame_number][i][0]), int(gt_data[frame_number][i][1]),int(gt_data[frame_number][i][2]), int(gt_data[frame_number][i][3])))
            else:
                f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(i, frame_number, 1, 1, int(gt_data[frame_number][i][0]), int(gt_data[frame_number][i][1]),int(gt_data[frame_number][i][2]), int(gt_data[frame_number][i][3])))
        f.close()
        label(data, annotation)



if __name__ == '__main__':
    '''
    data: input image folder
    annotation: json file using frame number as key, bbox coordinates as value
    output_file: path where you want to store output top file
    '''
    data = '/content/drive/My Drive/SocialDistance/data/CUHK/test'
    annotation = '/content/drive/My Drive/SocialDistance/data/CUHK_val.json'
    output_file = 'drive/My Drive/SocialDistance/data/CUHK/CUHK_groundtruth.top'

    top_generator = generate_top()
    top_generator.label(data, annotation,output_file)