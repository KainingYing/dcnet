# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head
from .basehoidetector import BaseHoiDetector


@DETECTORS.register_module(force=True)
class QPIC(BaseHoiDetector):
    """reimplement of CVPR 2021 QPIC
    https://arxiv.org/abs/2103.05399"""
    def __init__(self,
                 backbone,
                 hoi_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(QPIC, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        hoi_head.update(train_cfg=train_cfg)
        hoi_head.update(test_cfg=test_cfg)
        self.hoi_head = build_head(hoi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck.
        There is no neck here."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used fo computing network flops"""
        # TODO: future work
        pass

    def forward_train(self,
                      img,
                      img_metas,
                      gt_obj_bboxes):
        # NOTE the batched image size information may be useful
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        losses = self.hoi_head.forward_train(x, img_metas)
        return losses





