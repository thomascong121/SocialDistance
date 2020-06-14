import numpy as np
import mxnet as mx
from mxnet import gluon
from gluoncv import utils
from mxnet import nd
from gluoncv.utils import bbox_iou

class RepulsionLoss(gluon.Block):
    def __init__(self, iou_thresh = 0.5, sigma = 0.5, epo = 0.1, **kwargs):
        super(RepulsionLoss, self).__init__(**kwargs)
        self.iou_thresh = iou_thresh
        self.sigma = sigma
        self.epo = epo
    def Smooth_Ln(self, x, sigma):
        large = np.where(x > sigma)
        small = np.where(x <= sigma)

        large = x[large]
        small = x[small]

        large = np.sum((large-sigma)/(1-sigma) - np.log(1-sigma))
        small = np.sum(-np.log(1-small))

        return (large + small)
    def forward(self, cls_preds, box_preds, cls_targets, box_targets, loss = None):
        RepLoss = []
        all_box_gt = box_targets[0].asnumpy()
        all_box_pred = box_preds[0].asnumpy()
        for i in range(all_box_pred.shape[0]):
            #filter out all zero rows(mainly gt)
            nonzero_boxgt_index = np.where(all_box_gt[i][:,0] != all_box_gt[i][:,2])
            nonzero_boxpred_index = np.where(all_box_pred[i][:,0] != all_box_pred[i][:,2])

            nonzero_box_gt = all_box_gt[i][nonzero_boxgt_index][:,0:4]
            nonzero_box_pred = all_box_pred[i][nonzero_boxpred_index][:,0:4]

            #calculate iou
            _iou = bbox_iou(nonzero_box_pred, nonzero_box_gt)

            # select positive proposals
            pos_index = np.where(np.max(_iou, axis=1) >= self.iou_thresh)
            _iou = _iou[pos_index]
            #for each positive proposals keep its top two iou with targets
            sort_index = _iou.argsort(axis = 1)[:,-2:]
            iog = []
        for _i in range(len(sort_index)):
            tmp = _iou[_i, sort_index[_i]]
            iog.append(tmp)
        iog = np.array(iog)
        if iog.shape[0] == 0:
            RepGT = 0
            RepBo = 0
        else:
            #RepulsionGT
            RepGT = self.Smooth_Ln(iog[:,0], self.sigma)/iog.shape[0]
            #for each ground truth keep only the proposal with highest iou
            pos_gt_prop_index = np.argmax(_iou, axis=0)
            pos_gt_prop = np.array([nonzero_box_pred[pos_gt_prop_index], nonzero_box_pred[pos_gt_prop_index]])
            # RepulsionBox
            box_l = np.array([])
            total_iou = np.array([])
        for row in range(len(pos_gt_prop[0])-1):
            curr = pos_gt_prop[0][row].reshape(1,-1)
            rest = pos_gt_prop[1][row+1:]
            _bbox_iou = bbox_iou(curr, rest)
            box_l = np.hstack((box_l, [self.Smooth_Ln(_bbox_iou, self.sigma)]))
            total_iou = np.hstack((total_iou, [np.sum(_bbox_iou)]))
        RepBo = np.sum(box_l) / (np.sum(total_iou) + self.epo)
        RepLoss.append(RepGT + RepBo)
    RepLoss = [nd.array(RepLoss, ctx=mx.gpu(0))]
    if loss:
        sum_loss, cls_loss, box_loss = loss(cls_preds, box_preds, cls_targets, box_targets)#TODO:YOLO-VERSION
        return nd.add(RepLoss[0], sum_loss[0]), cls_loss, box_loss
    else:
        return RepLoss, 0,0