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
        for i in range(self.npics):
            ret,frame = cap.read()
            print(type(frame),frame.shape)
            cv2.imwrite(save_path + '/%d.jpg'%i,frame)
            
            img = cv2.imread(save_path + '/%d.jpg'%i)
            h,w,c = img.shape
            h = int(h*0.2)
            w = int(w*0.5)
            dim = (256,256)
            reshaped = cv2.resize(img,dim)
            cv2.imshow('img',reshaped)
            cv2.waitKey()
        print('read {0} out of {1} and save to {2}'.format(self.npics,frames,save_path))


if __name__ == '__main__':
    # reader = VideoRead(1)
    # reader.frame_capture('../data/TownCentreXVID.avi','../video_frames')
    # reader.frame_to_video('../video_frames')
    img = cv2.imread('../video_frames/0.jpg')
    plt.imshow(img)
    plt.show()
