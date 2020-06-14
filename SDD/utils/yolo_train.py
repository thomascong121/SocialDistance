import time
import mxnet as mx
from mxnet import autograd, gluon
from gluoncv import autograd
import gluoncv as gcv
from skimage import io
from tqdm import tqdm
from mxnet import nd
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from .DatasetGenerator import DatasetGenerator
from .GreedyRepulsion import RepulsionLoss

class YOLO_custom_train:
    def __init__(self, datasetname, data, annotation):
        self.net = gcv.model_zoo.get_model('yolo3_darknet53_custom', classes=train_gt.classes, pretrained_base=False, transfer='voc')
        try:
            a = mx.nd.zeros((1,), ctx=mx.gpu(0))
            self.ctx = [mx.gpu(0)]
        except:
            self.ctx = [mx.cpu()]
        self.data_gen = DatasetGenerator(datasetname)
        self.train_gt, self.val_gt, self.train_lst_dataset, self.val_lst_dataset = self.data_gen(data, annotation)
        self.width, self.height = self.get_width_height(io.imread(self.train_gt[0][0]).shape[:2])

    def get_width_height(self, img_shape, short_side = 512):
        short = min(img_shape)
        height, weight = img_shape[0]*(short_side/short), img_shape[1]*(short_side/short)
        return int(weight), int(height)

    def get_dataloader(self, net, train_dataset, val_dataset, width, height, batch_size, num_workers):
        from gluoncv.data.batchify import Tuple, Stack, Pad
        from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform, YOLO3DefaultValTransform

        # use fake data to generate fixed anchors for target generation    
        train_batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net)),
            batch_size, True, batchify_fn=train_batchify_fn, last_batch='rollover', num_workers=num_workers)
        
        val__batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(width, height)),
            batch_size, True, batchify_fn=val__batchify_fn, last_batch='rollover', num_workers=num_workers) 
        
        return train_loader, val_loader

    def validate(self, net, val_data, ctx, eval_metric):
        """Test on validation dataset."""
        eval_metric.reset()
        # set nms threshold and topk constraint
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        net.hybridize(static_alloc=True, static_shape=True)
        for batch in val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return eval_metric.get()

    def train(self, net, train_loader, val_loader, eval_metric, ctx, epochs = 10, save_path = './', optimizer_parameter = {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9}, seed = 0):
        utils.random.seed(seed)
        net.collect_params().reset_ctx(ctx)
        trainer = gluon.Trainer(
            net.collect_params(), 'sgd',
            {'learning_rate': optimizer_parameter['learning_rate'], 'wd': optimizer_parameter['wd'], 'momentum': optimizer_parameter['momentum']})

        mbox_loss = gcv.loss.SSDMultiBoxLoss(lambd = 0.5)
        Repulsion = RepulsionLoss()#gcv.loss.SSDMultiBoxLoss()
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')

        best_mean_ap = float('-inf')
        for epoch in range(epochs):
            obj_metrics.reset()
            center_metrics.reset()
            scale_metrics.reset()
            cls_metrics.reset()

            tic = time.time()
            btic = time.time()

            # print(epochs, optimizer_parameter['lr_decay_step'], epoch % optimizer_parameter['lr_decay_step'] == 0)
            if 'lr_decay' in optimizer_parameter and 'lr_decay_step' in optimizer_parameter\
            and epoch != 0 and epoch % optimizer_parameter['lr_decay_step'] == 0:
                old_lr = trainer.learning_rate
                new_lr = trainer.learning_rate * optimizer_parameter['lr_decay']
                trainer.set_learning_rate(new_lr)
                print(f'The learning rate from {old_lr} change to {new_lr}')


            net.hybridize(static_alloc=True, static_shape=True)
            qbar = tqdm(train_loader)
            qbar.desc = f'Epoch {epoch + 1}'
            for i, batch in enumerate(qbar):
                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
                cls_target = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)
                sum_losses = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    autograd.backward(sum_losses)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(batch_size)
                obj_metrics.update(0, obj_losses)
                center_metrics.update(0, center_losses)
                scale_metrics.update(0, scale_losses)
                cls_metrics.update(0, cls_losses)
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                qbar.set_postfix({'Batch': i, name1: loss1, name2: loss2, name3: loss3, name4: loss4})
                btic = time.time()
            map_name, mean_ap = validate(net, val_loader, ctx, eval_metric)
            print(f'{map_name}: {mean_ap}')
            if best_mean_ap <= mean_ap[-1]:
                net.save_parameters(f'yolo_best_{epoch}.params')
                best_mean_ap = mean_ap[-1]
    def __call__(self):
        train_loader, val_loader = get_dataloader(self.net, self.train_lst_dataset, self.val_lst_dataset, self.width, self.height, 5, 0)
        eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_gt.classes)
        optimizer_parameter = {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9, 'lr_decay': 0.5, 'lr_decay_step': 4}
        self.train(net, train_loader, val_loader, eval_metric, ctx = self.ctx, optimizer_parameter = optimizer_parameter, epochs = 25)