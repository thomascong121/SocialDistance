import json
import os, zipfile
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from gluoncv.utils import viz
from skimage import io
from tqdm import tqdm
import glob
'''
Generate lst files for both Oxford and Caltech dataset
'''

class LstGenerator_Oxford:
    def __init__(self, images_path, gtboxes, classes = ['person']):
        self.images = sorted(glob.glob(f'{images_path}/*'), key = lambda x: int(x.split('/')[-1].split('.')[0]))
        self.gtboxes = gtboxes
        self.classes = classes

    def imshow(self, index):
        path, gtboxes = self[index]
        img = io.imread(path)
        ax = viz.plot_bbox(img, bboxes=gtboxes, labels = np.zeros(len(gtboxes)), class_names=self.classes)
        plt.show()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        num = int(self.images[index].split('/')[-1].split('.')[0])
        return self.images[index], self.gtboxes[num]

    def _write_line(self, img_path, im_shape, boxes, ids, idx):
        h, w, c = im_shape
        # for header, we use minimal length 2, plus width and height
        # with A: 4, B: 5, C: width, D: height
        A = 4
        B = 5
        C = w
        D = h
        # concat id and bboxes
        labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
        # normalized bboxes (recommanded)
        labels[:, (1, 3)] /= float(w)
        labels[:, (2, 4)] /= float(h)
        # flatten
        labels = labels.flatten().tolist()
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = [str(x) for x in labels]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line

    def generate_lst_file(self, filename, filepath = './'):
        img_shape = None
        with open(f'{filepath}/{filename}.lst', 'w') as fw:
            for i in tqdm(range(len(self))):
                img_path, gtboxes = self[i]
                if img_shape is None:
                    img_shape = io.imread(img_path).shape
                line = self._write_line(img_path, img_shape, gtboxes, np.zeros(len(gtboxes)), i)
                fw.write(line)

class LstGenerator_Caltech:
    def __init__(self, caltech_datapath, annotation, start = 0, end = 0, classes = ['person']):
        all_files = sorted(glob.glob(f'{caltech_datapath}/*'), key = lambda x: x.split('/')[-1])
        end = len(all_files) if end == 0 else end
        self.sets = list(filter(lambda x:os.path.isdir(x), all_files))[start:end]
        json_file = open(annotation) 
        self.annotation = json.load(json_file)
        self.classes = classes

    def imshow(self, index):
        video_path, sets_number = self[i]
        video_number = os.path.basename(video_path)
        for f in sorted(glob.glob(f'{video_path}/*')):
          f_number = os.path.basename(f)
          gt = self.annotation[sets_number][video_number]['frames'][f_number]

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, index):
        return sorted(glob.glob(f'{self.sets[index]}/*')), os.path.basename(self.sets[index])

    def _write_line(self, img_path, im_shape, boxes, ids, idx):
        h, w, c = im_shape
        # for header, we use minimal length 2, plus width and height
        # with A: 4, B: 5, C: width, D: height
        A = 4
        B = 5
        C = w
        D = h
        # concat id and bboxes
        labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
        # normalized bboxes (recommanded)
        labels[:, (1, 3)] /= float(w)
        labels[:, (2, 4)] /= float(h)
        # flatten
        labels = labels.flatten().tolist()
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = [str(x) for x in labels]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line

    def generate_lst_file(self, filename, filepath = './'):
        img_shape = None
        with open(f'{filepath}/{filename}.lst', 'w') as fw:
            for i in tqdm(range(len(self))):
                video_path, sets_number = self[i]
                for v in video_path:
                  video_number = os.path.basename(v)
                  for f in sorted(glob.glob(f'{v}/*')):
                    f_number = os.path.basename(f).split('.')[0]
                    if img_shape is None:
                        img_shape = io.imread(f).shape
                    gt = self.annotation[sets_number][video_number]['frames'][f_number]
                    gt_box = []
                    for g in gt:
                      gt_box.append([g['pos'][0], g['pos'][1], g['pos'][0]+g['pos'][2], g['pos'][1]+g['pos'][3]])
                    gt_box = np.array(gt_box)
                    line = self._write_line(f, img_shape, gt_box, np.zeros(len(gt_box)), i)
                    fw.write(line)