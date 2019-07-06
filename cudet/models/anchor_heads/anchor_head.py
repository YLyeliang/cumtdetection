import numpy as np
import torch
import torch.nn as nn
from ..utils import normal_init

from cudet.core import (AnchorGenerator,multi_apply)
from ..builder import build_loss




class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8,16,32],
                 anchor_ratios=[0.5,1.0,2.0],
                 anchor_strides=[4,8,16,32,64],
                 anchor_base_sizes=None,
                 target_means=(.0,.0,.0,.0),
                 target_stds=(1.0,1.0,1.0,1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss',beta=1.0/9.0,loss_weight=1.0)):
        super(AnchorHead,self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels =feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes  = list(          # base anchor sizes: a proportion of feat map with respect to input
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid',False)    # if None, set default False.
        self.sampling = loss_cls['type'] not in ['FocalLoss','GHMC']

        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes -1
        else:
            self.cls_out_channels = num_classes

        # build losses
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        # generate anchors by using AnchorGenerator, anchor_base are anchor_stride e.g 4 8 16 32 64
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales,anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)


    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels,1)
        self.conv_reg = nn.Conv2d(self.feat_channels,self.num_anchors*4, 1)

    def init_weights(self):
        normal_init(self.conv_cls,std=0.01)
        normal_init(self.conv_reg,std=0.01)

    def forward_single(self,x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single,feats)

    def get_anchors(self,featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)


        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i],self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags= []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h,feat_w =featmap_sizes[i]
                h,w,_ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h/anchor_stride)),feat_h)
                valid_feat_w = min(int(np.ceil(w/anchor_stride)),feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h,feat_w),(valid_feat_h,valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list,valid_flag_list

    def loss_single(self,cls_score, bbox_pred,labels,label_weights,
                    bbox_targets,bbox_weights,num_total_samples,cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0,2,3,1).reshape(-1,self.cls_out_channels)    # NCWH
        loss_cls = self.loss_cls(
            cls_score,labels,label_weights,avg_factor=num_total_samples)
        # regression loss
        bbox_tarets = bbox_targets.reshape(-1,4)
        bbox_weights = bbox_weights.reshape(-1,4)
        bbox_pred = bbox_pred.permute(0,2,3,1).reshape(-1,4)
        loss_bbox = self.loss_bbox(
            bbox_pred,bbox_tarets,bbox_weights,avg_factor=num_total_samples
        )
        return loss_cls,loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] # NCHW, get list of H,W
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes,img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
