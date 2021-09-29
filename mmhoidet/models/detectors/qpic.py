# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head
from .basehoidetector import BaseHoiDetector
from mmhoidet.core.hoi import hoi2result


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
                      gt_sub_bboxes,
                      gt_obj_bboxes,
                      gt_obj_labels,
                      gt_verb_labels,
                      **kwargs):
        # NOTE the batched image size information may be useful
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        losses = self.hoi_head.forward_train(x, img_metas, gt_sub_bboxes, gt_obj_bboxes,
                                             gt_obj_labels, gt_verb_labels, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.hoi_head.simple_test(
            feat, img_metas, rescale=rescale)
        hoi_results = [
            hoi2result(instance_labels, verb_scores, bboxes, sub_ids, obj_ids, valid_hois=self.valid_hois)
            for instance_labels, verb_scores, bboxes, sub_ids, obj_ids in results_list
        ]
        return hoi_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        hoi_results = [
            hoi2result(instance_labels, verb_labels, bboxes, sub_ids, obj_ids)
            for instance_labels, verb_labels, bboxes, sub_ids, obj_ids in results_list
        ]
        return hoi_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels



