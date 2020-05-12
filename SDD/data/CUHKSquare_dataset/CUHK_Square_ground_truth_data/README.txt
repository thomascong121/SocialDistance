This directory contains associated ground truth data for the CUHK Square dataset.

The CUHK Square dataset contains a single video clip, and is available for download at http://www.ee.cuhk.edu.hk/~xgwang/CUHKSquare.html.

``train_ground_truth_data.mat'' contains training data in our 2012 work [1]. The ground truth data itself is not used in the training process, but just for evaluation. Read the paper for more detail.

``test_ground_truth_data.mat'' contains testing data in our 2012 work.

Upon loaded in MATLAB, a new struct variable will be available as ``ground_truth_data'', which contain two fields: ``list'' and ``imgsize''.

``imgsize'' is the actual image size (rather than the raw frames stripped out directly from the clips). In our experiment, the raw frames are resized into 2 times, and therefore the image size is 1152 by 1440.

``list'' is the cell array containing ground truth bounding boxes. Each entry contains two fields: ``imgname'' and ``bbox''.

``imgname'' is given in the form ``Culture_Square_xxxxx.png'', where xxxxx corresponds the frame number (step is 200 frames). The image files are not provided due to space issue, but you can always strip out the frames and resize them to the proper size by yourself.

``bbox'' contains an N by 4 matrix, in which N is the number of ground truths in that frame. Each row is given in the format (x, y, width, height).

Example:

See the following pseudo code for an example demonstrating how to use the ground truth data to draw the rectangles on pedestrians:

load ground_truth_data_train.mat;
for i = 1:numel(ground_truth_data.list)
    img = imread(ground_truth_data.list{i}.imgname);
    bbox = ground_truth_data.list{i}.bbox;
    for j = 1:size(bbox, 1)
        img = draw_rectangle_on_image_xywh(img, bbox(j, :)); % this function is a pseudo function
    end;
    imshow(img);
end;

Reference:
If you use the CUHK Square dataset (not including the ground truth data), please cite properly according to http://www.ee.cuhk.edu.hk/~xgwang/CUHKSquare.html.
If you use this ground truth data, please cite as

[1] Meng Wang, Wei Li and Xiaogang Wang. Transferring a Generic Pedestrian Detector Towards Specific Scenes. IEEE Conference on Computer Vision and Pattern Recognition 2012, Providence, Rhode Island, USA.
