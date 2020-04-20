import cv2
import numpy
'''
read video and generate frames
'''
class VideoRead:
    def __init__(self,npics=0):
        self.npics = npics
    def frame_capture(self,video,save_path):
        '''read video and capture frames then save to path'''
        cap = cv2.VideoCapture(video)
        print('===vide info===')
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('total frames: ',frames)
        for i in range(10):
            ret,frame = cap.read()
            cv2.imwrite(save_path + '/%d.png'%i,frame)
        print('read {0} out of {1} and save to {2}'.format(self.npics,frames,save_path))

if __name__ == '__main__':
    reader = VideoRead(10)
    reader.frame_capture('./TownCentreXVID.avi','./video_frames')