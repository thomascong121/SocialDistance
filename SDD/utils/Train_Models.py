from .ssd_train import SSD_custom_train
from .yolo_train import YOLO_custom_train

'''
Custom model training with provided datasets:
for both SSD and YOLO, params include:
    datasetname: 'Oxford' or 'Caltech'
    data: path to video or image folder
    annotation: annotation files for datasets in .json or .top
'''

class Train_Custom:
    def __init__(self, model, datasetname, data, annotation):
        if model == 'SSD':
            SSD_trainer = SSD_custom_train(datasetname, data, annotation)
            SSD_trainer()
        if model == 'YOLO':
            YOLO_trainer = YOLO_custom_train(datasetname, data, annotation)
            YOLO_trainer()
        #TODO: Support more models