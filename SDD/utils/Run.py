#main
# from Transformer import Bird_eye_view_Transformer
import cv2
import mxnet as mx
from .View_Transformer import Bird_eye_view_Transformer
from .Detector import VideoDetector, ImageDetector
from .Models import Bbox_detector
class Detect:
    def __init__(self, keypoints = [(1175,  189), (1574,  235), (364,  694), (976,  831)], \
        keypoints_birds_eye_view = [(900,  300), (1200,  300), (900,  900), (1200,  900)], \
        actual_length = 10,  actual_width = 5, pretrained_models = 'yolo_v3'):
        '''
        main function used for detection and labelling

        Parameters
        ----------
        keypoints: selected key points from first frame of the input video
        keypoints_birds_eye_view: mapping location of keypoints on the bird-eye view image
        actual_length: actual length in real-world
        actual_width: actual width in real-world
        pretrained_models: selected pretrained models -- yolo_v3, ssd, faster_rcnn, center_net
        '''
        self.keypoints = keypoints
        self.keypoints_birds_eye_view = keypoints_birds_eye_view
        self.actual_length = actual_length
        self.actual_width = actual_width
        self.pretrained_models = pretrained_models
        self.transformer = Bird_eye_view_Transformer(self.keypoints, self.keypoints_birds_eye_view, self.actual_length, self.actual_width, multi_pts = False)
    
    def __call__(self, save_path, video = True, threshold = 0.7, need_view_tranformer = False, interval = 1):
        transformer = self.transformer if need_view_tranformer == True else None
        detect_model = Bbox_detector(self.pretrained_models, transformer, mx.gpu())
        detector = VideoDetector(detect_model, threshold = threshold, save_path = save_path, interval = interval) if video else ImageDetector(detect_model, threshold = threshold, save_path = save_path)
        return detector
