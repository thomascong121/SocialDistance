# SocialDistance
Keep safe social distance is considered as an effective way of avoiding spreading of coronavirus. Our SocialDistance module __SDD__ is a lightweight package which provides an implementation of utlizing deep learning models for monitoring safe social distance.

# Demo
[Watch the demo video](https://www.youtube.com/watch?v=1s46BJJj6rw&t=5s)

# Dataset
We use the video clip collected from [OXFORD TOWN CENTRE](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html) dataset and made the above demo video.

# Supported Models
We have tested our model using Faster-RCNN, YOLO-v3 and SSD, based on the performance of each model, we have chosen YOLO-v3 as our default model

All our models are pretrained models from [Gluno CV Tookit](https://github.com/dmlc/gluon-cv)

# Installation
You may be able to obtain the latest version our model from:
```
pip install SDD
```

# Usage
After Successfully installed SocialDistance, you can use it for detection by:
```
from SocialDistance.utils.Run import Detect
detect = Detect()
#you may want to give an image as input to check the validity of bird-eye view transformation
detect(image)
```
If no arguments is given, our model will run using the default data collected from 'OXFORD TOWN CENTRE' dataset, otherwise you may want to specify arguments expicitly:
```
from SocialDistance.utils.Run import Detect
detect = Detect(video_path, video_save_path, keypoints, keypoints_birds_eye_view, actual_length, actual_width, pretrained_models)
#you may want to give an image as input to check the validity of bird-eye view transformation
detect(image)
```
> Parameters
> ----------
- **video_path**: input path of video
- **video_save_path**: output path of labelled video
- **keypoints**: selected key points from first frame of the input video
- **keypoints_birds_eye_view**: mapping location of keypoints on the bird-eye view image
- **actual_length**: actual length in real-world
- **actual_width**: actual width in real-world
- **pretrained_models**: selected pretrained models
# Reference
1. Landing AI 16 April 2020, Landing AI Creates an AI Tool to Help Customers Monitor Social Distancing in the Workplace, accessed 19 April 2020, <https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/>


