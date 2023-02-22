# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.models.builder import build_backbone, build_neck, build_head, DETECTORS  # noqa
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import BaseDetector


@DETECTORS.register_module()
class DCNet(BaseDetector):
    """
    Accurate features for human interaction detection
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,  # use for detect instances
                 hoi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DCNet, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)

        # if self.

        # hoi_head.update(train_cfg=train_cfg)
        # hoi_head.update(test_cfg=test_cfg)
        # self.hoi_head = build_head(hoi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck.
        There is no neck here."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      **kwargs):
        losses = {}

        # NOTE the batched image size information may be useful
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)

        losses_detr, embeds, memory = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, **kwargs)

        embeds, memory = embeds[0], memory[0]
        losses_hoi = self.hoi_head.forward_train(embeds, memory, gt_)

        losses.update(losses_detr)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError
