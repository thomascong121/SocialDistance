# SocialDistance
Keep safe social distance is considered as an effective way of avoiding spreading of coronavirus. Our SocialDistance module __SDD__ is a lightweight package which provides an implementation of utlizing deep learning models for monitoring safe social distance.

# Demo
[Watch the demo video](https://www.youtube.com/watch?v=1s46BJJj6rw&t=5s)

# Dataset
We use the video clip collected from [OXFORD TOWN CENTRE](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html) dataset and made the above demo video.

# Supported Models
We have tested our model using YOLO-v3 and SSD, based on the performance of each model, we have chosen YOLO-v3 as our default model

All our pretrained models are selected from [Gluno CV Tookit](https://github.com/dmlc/gluon-cv)

# Installation
You may be able to obtain the latest version our model from:
```
pip install SDD
```

# How to Use
After Successfully installed SocialDistance, you can use it for detection by:
```
import SDD
import mxnet as mx
from SDD.utils.Run import Detect
detect = Detect(pretrained_models = 'yolo_v3')
detector = detect(save_path = output_path_image, video = False, device = mx.cpu())
_ = detector(file_path)
```

> Parameters
> ----------
- **pretrained_models**: str, Currently, we provided two pretrained models: 'yolo_v3' and 'ssd'
- **save_path**: str, Path where you want to save the output video/ images
- **video**: boolean, If your input is a video, set this parameter as True, if your input is a set of images, set this parameter as False
- **device**: mx.cpu() or mx.gpu()
- **file_path**: str, Input path of your video or image folder
# Reference
1. Landing AI 16 April 2020, Landing AI Creates an AI Tool to Help Customers Monitor Social Distancing in the Workplace, accessed 19 April 2020, <https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/>


