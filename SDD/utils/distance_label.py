import numpy as np
import scipy.io as scio
import cv2, os
class GroundTruthDetections:
    def __init__(self, fname):
        base, ext = os.path.splitext(fname)
        if ext == '.mat':
            mat_file = scio.loadmat(filename) 
            

        else:
            self.all_dets = np.loadtxt(fname, delimiter = ',')

    def read(self,filename):
        data = scio.loadmat(filename)
        print(data)
        
    def show(self, video_path = None, images_path = None):
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print('===vide info===')
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print('total frames: ',frames)
        elif images_path:
            frames = os.listdir(images_path)
        else:
            raise ValueError('Lack of video/image path')

        gt = [x for x in range(0,len(self.all_dets),len(self.all_dets)//100)]
        for i in range(len(self.all_dets)):#len(self.all_dets)
            ret,frame = cap.read()
            frame_gt = self.all_dets[self.all_dets[:,1] == i]
            if i in gt:
                print('total {0} bbox in {1} frame'.format(len(frame_gt),i))
                for j in range(len(frame_gt)):
                    bodytopleft_x, bodytopleft_y = int(frame_gt[j][-4]), int(frame_gt[j][-3])
                    bodybottomright_x, bodybottomright_y = int(frame_gt[j][-2]), int(frame_gt[j][-1])

                    # # headright_x, headright_y = int(frame_gt[j][4]), int(frame_gt[j][5])
                    # # headleft_x, headleft_y = int(frame_gt[j][6]), int(frame_gt[j][7])
                    # center_1 = ((bodytopleft_x + bodybottomright_x)//2, (bodytopleft_y + bodybottomright_y)//2)
                    # # dist = float('inf')
                    # cv2.rectangle(frame, (bodytopleft_x,  bodytopleft_y), (bodybottomright_x, bodybottomright_y), (0, 255, 0), 2)
                    # for i in range(j+1, len(frame_gt)):
                    #     bodytopleft_x, bodytopleft_y = int(frame_gt[i][-4]), int(frame_gt[i][-3])
                    #     bodybottomright_x, bodybottomright_y = int(frame_gt[i][-2]), int(frame_gt[i][-1])
                    #     cv2.rectangle(frame, (bodytopleft_x,  bodytopleft_y), (bodybottomright_x, bodybottomright_y), (0, 255, 0), 2)
                    #     center_2 = ((bodytopleft_x + bodybottomright_x)//2, (bodytopleft_y + bodybottomright_y)//2)
                    #     distance = (center_1[0] - center_2[0])**2 + (center_1[1] - center_2[1])**2
                    #     if distance <= 25000:
                    #         cv2.line(frame, center_1, center_2, (0, 255, 0), 2)
                    #         middle = ((center_1[0] + center_2[0])//2, (center_1[1] + center_2[1])//2)
                    #         cv2.putText(frame, 'dist: {0}'.format(distance), middle, cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

                    cv2.rectangle(frame, (bodytopleft_x,  bodytopleft_y), (bodybottomright_x, bodybottomright_y), (0, 255, 0), 2)
                    cv2.putText(frame, 'ID: {0}'.format(frame_gt[j][0]), ((bodytopleft_x + bodybottomright_x)//2, (bodytopleft_y + bodybottomright_y)//2), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                dim = (900,732)
                reshaped = cv2.resize(frame, dim)
                cv2.imshow('img', reshaped)
                cv2.waitKey()


if __name__ == '__main__':
    GT = GroundTruthDetections('../data/TownCenter_dataset/TownCentre-groundtruth.top')#
    GT.show('../data/TownCenter_dataset/TownCentreXVID.avi')
    # GT.show()
    # GT.read('../data/mall_dataset/mall_gt.mat')






