from gluoncv.data import LstDetection
from .Video2images import Video2ImageSequence, Top2Dict
from .LstGenerator import LstGenerator_Oxford, LstGenerator_Caltech
'''
Generate datasets used for training and validation
Note: If the dataset is form of video, it is necessary to first transform it into images
'''
class DatasetGenerator:
    def __init__(self, dataset):
        self.dataset = dataset
    def __call__(self, data, annotation):
        if self.dataset == 'Oxford':
            images_generator = Video2ImageSequence()
            gtboxes = Top2Dict(annotation)
            #Generate Training set
            images_generator(data, './TrainImages', end = 3501)
            train_gt = LstGenerator_Oxford('./TrainImages', gtboxes)
            train_gt.generate_lst_file('train')
            #Generate Validation set
            images_generator(data, './ValImages', start = 3501, end = 4501)
            val_gt = LstGenerator_Oxford('./ValImages', gtboxes)
            val_gt.generate_lst_file('val')
            train_lst_dataset = LstDetection('train.lst')
            val_lst_dataset = LstDetection('val.lst') 
            return train_gt, val_gt, train_lst_dataset, val_lst_dataset

        if self.dataset == 'Caltech':
            caltech_datapath = data
            annotation = annotation
            train_gt = LstGenerator_Caltech(caltech_datapath, annotation, start = 0, end = 6)
            train_gt.generate_lst_file(caltech_datapath + '/Caltech_train')
            val_gt = LstGenerator_Caltech(caltech_datapath, annotation, start = 6,end = 11)
            train_gt.generate_lst_file(caltech_datapath + '/Caltech_val')
            train_lst_dataset = LstDetection(caltech_datapath + '/Caltech_train.lst')
            val_lst_dataset = LstDetection(caltech_datapath + '/Caltech_val.lst')
            return train_gt, val_gt, train_lst_dataset, val_lst_dataset