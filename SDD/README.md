# SocialDistance
Keep safe social distance is considered as an effective way of avoiding spreading of coronavirus. Our SocialDistance module is a lightweight package which provides an implementation of utlizing deep learning models for monitoring safe social distance.

# Demo
[Watch the demo video](https://www.youtube.com/watch?v=1s46BJJj6rw&t=5s)

# Dataset
We use the video clip collected from [OXFORD TOWN CENTRE](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html) dataset and made the above demo video.

# Supported Models
We have tested our model using Faster-RCNN, YOLO-v3 and SSD, based on the performance of each model, we have chosen YOLO-v3 as our default model

All pretrained models are from [Gluno CV Tookit](https://github.com/dmlc/gluon-cv). Besides, we have trained a YOLO-v3 and SSD using data from [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), the trained weights will be provided on request. A demo on `SDD/Demo` shows how the training of YOLO-v3 using Caltech dataset was done.

# Installation
You may be able to obtain the latest version our model from:
```
pip install SDD==0.2.2.7
pip install gluoncv
pip install mxnet-cu101
```

# Usage
After Successfully installed SocialDistance, you can use it for detection by:
```
from SocialDistance.utils.Run import Detect
detect = Detect(keypoints, keypoints_birds_eye_view, actual_length, actual_width, pretrained_models, trained)
detector = detect(save_path, video = True, threshold = 0.5, need_view_tranformer = False, device = mx.gpu())
out, TP, FP, TN, FN, extra = detector(video_path, image_groundTruth)

#Example:
video_path = path to your video
img_path = path to your video framesDrive/SocialDistance/data/CUHK/test'#'/content/drive/My Drive/SocialDistance/data/Mall'
output_path_video = path to your output video
image_groundTruth = path to your ground truth file

# YOLOv3 test
detect = Detect(pretrained_models = 'yolo_v3')
detector = detect(save_path = output_path_video, video = False, need_view_tranformer = False, device = mx.cpu())
out, TP, FP, TN, FN, extra = detector(img_path, image_groundTruth)
```
Running the above code will generate a lablled video. Moreover, `image_groundTruth` is a file contains the labelled (safe or unsafe) data. We have mannually labelled all three datasets and the `image_groundTruth` for all three dataset will be provided on request. 

# Support
For any issues, please contact me at:
Thomas Cong: thomascong@outlook.com,
Zhichao Yang: yzcwansui@outlook.com
