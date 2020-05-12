import numpy as np
import scipy.io as scio
import cv2
class GroundTruthDetections:
    def __init__(self, fname = None):
        self.all_dets = np.loadtxt(fname, delimiter = ',') if fname else None
    def read(self,filename):
        data = scio.loadmat(filename)
        print(data)
        
    def show(self, video):
        cap = cv2.VideoCapture(video)
        print('===vide info===')
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('total frames: ',frames)

        for i in range(10):#len(self.all_dets)
            ret,frame = cap.read()
            frame_gt = self.all_dets[self.all_dets[:,1] == i]
            for j in frame_gt:

                bodyleft_x, bodyleft_y = int(j[-4]), int(j[-3])
                bodyright_x, bodytomright_y = int(j[-2]), int(j[-1])

                headright_x, headright_y = int(j[4]), int(j[5])
                headleft_x, headleft_y = int(j[6]), int(j[7])

                cv2.circle(frame, ((bodyleft_x + bodyright_x)//2, (bodyleft_y + bodytomright_y)//2), 2, (0, 255, 0), 2)
                # dim = (512,512)
                # reshaped = cv2.resize(frame,dim)
            cv2.imshow('img',frame)
            cv2.waitKey()


if __name__ == '__main__':
    GT = GroundTruthDetections()#'../data/TownCentre-groundtruth.top'
    # GT.show('../data/TownCentreXVID.avi')
    # GT.show()
    GT.read('../data/mall_dataset/mall_gt.mat')