import cv2
import numpy
import os
from matplotlib import pyplot as plt
'''
read video and generate frames
'''
class VideoRead:
    def __init__(self,npics=0):
        self.npics = npics
    def frame_capture(self,video,save_path):
        '''read video and capture frames then save to path'''
        print(video)
        cap = cv2.VideoCapture(video)
        print('===vide info===')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('total frames: ',frames)
        print('frames size: ','1080*1920*3')
        print('fps :',fps)
        count = 0
        need = [x for x in range(0, 4501, 45)]
        for i in range(5000):
            ret,frame = cap.read()
            if i in need:
                cv2.imwrite(save_path + '/%d.jpg'%i,frame)
                count += 1
                
                # img = cv2.imread(save_path + '/%d.jpg'%i)
                # h,w,c = img.shape
                # h = int(h*0.2)
                # w = int(w*0.5)
                # dim = (256,256)
                # reshaped = cv2.resize(img,dim)
                # cv2.imshow('img',reshaped)
                # cv2.waitKey()
        print('read {0} out of {1} and save to {2}'.format(count,frames,save_path))


if __name__ == '__main__':
    reader = VideoRead(4501)
    reader.frame_capture('/Users/congcong/Desktop/SocialDistanceDetector/SDD/data/TownCenter_dataset/TownCentreXVID.avi',\
        '/Users/congcong/Desktop/SocialDistanceDetector/SDD/video_frames')
    # plt.imshow(img)
    # plt.show()
