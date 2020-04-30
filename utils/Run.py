#main
import gluoncv
from gluoncv import model_zoo, data, utils
from Transformer import Bird_eye_view_Transformer
from matplotlib import pyplot as plt

keypoints = [(1175,  189), (1574,  235), (364,  694), (976,  831)]
keypoints_birds_eye_view = [(900,  300), (1200,  300), (900,  900), (1200,  900)]
actual_length = 10
actual_width = 5
img_path = '/content/drive/My Drive/SocialDistance/video_frames/0.png'
video_path = '/content/drive/My Drive/SocialDistance/data/TownCentreXVID.avi'


print('Testing transformer.....')
img = cv2.imread(img_path)
transformer = Bird_eye_view_Transformer(keypoints, keypoints_birds_eye_view, actual_length, actual_width, multi_pts = False)
transformer.imshow(img)
print('Starting detecting.....')
pretrained_models = 'yolo3_darknet53_voc'
detect_model = Model_Zoo(pretrained_models, transformer, mx.gpu())

detector = Detector(detect_model, save_path = '/content/drive/My Drive/SocialDistance/outputs', interval = 1)
detector(video_path)
