# SocialDistance
Keeping safe social distance is considered as an effective way of avoiding spreading of coronavirus. Our SocialDistance module is a lightweight package, which provides a deep learning method for monitoring safe social distance.

# Demo
[Watch the demo video](https://www.youtube.com/watch?v=1s46BJJj6rw&t=5s)

# Dataset
We use the video clip collected from [OXFORD TOWN CENTRE](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html) dataset and made the above demo video. 

We trained our detectors using data from [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/). Moreover, we tested our model on [OXFORD TOWN CENTRE](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html), [CUHK Square](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_square.html) and [Mall](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)

# Supported Models
We have tested our model using Faster-RCNN, CenterNet, YOLO-v3 and SSD. Based on the performance of each model, we have chosen YOLO-v3 as our default model

All pre-trained models are from [Gluno CV Tookit](https://github.com/dmlc/gluon-cv). Besides, we have trained YOLO-v3 and SSD using data from [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/). The trained model parameters will be available upon request. A demo on `SDD/Demo` shows how the training of YOLO-v3 using Caltech dataset was done.

# Installation
You may be able to obtain the latest version of our model from:
```
pip install SDD==0.2.2.7
pip install gluoncv
pip install mxnet-cu101
```

# Usage
After successfully installing SDD, you can use it for detection by:
```
from SocialDistance.utils.Run import Detect
video_path = path to your video
img_path = path to your video frames
output_path_video = path to your output video
image_groundTruth = path to your ground truth file

# YOLOv3 test
detect = Detect(pretrained_models = 'yolo_v3')
detector = detect(save_path = output_path_video, video = False, need_view_tranformer = False, device = mx.cpu())
out, TP, FP, TN, FN, extra = detector(img_path, image_groundTruth)
```
Running the above code will generate a labelled video. 
`image_groundTruth` is a `.top` file that contains the labels of social distance. We have manually labeled all three datasets and in each row of the file, the second element indicates the frame number, the third element shows the labels of social distance (0 = unsafe and 1 = safe) and the last four elements are the bounding box coordinates. The `.top` file of Oxford TownCenter, CUHK and Mall dataset are available under the data folder.

# Support
For any issue, please contact us at:  
Thomas Cong: thomascong@outlook.com,  
Zhichao Yang: yzcwansui@outlook.com

# Citation
If you find the code/data is useful, please cite our paper:
```
@inproceedings{2020SSD,
    title={Towards Enforcing Social Distancing Regulations with Occlusion-Aware Crowd Detection},
    booktitle={International Conference on Control, Automation, Robotics and Vision},
    author={Cong Cong and Zhichao Yang and Yang Song and Maurice Pagnucco},
    year={2020}
}
```
