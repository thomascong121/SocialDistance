% Example

load('mall_gt.mat');

img_name = './frames/seq_%.6d.jpg';

img_index = 970;

im = imread(sprintf(img_name,img_index));

XY=frame{img_index}.loc;

imshow(im); hold on;
plot(XY(:,1),XY(:,2),'r*');