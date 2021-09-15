# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import HOI_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@HOI_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually.
    Modified form `mmdet.bbox.samplers.PseudoSampler`"""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, sub_bboxes, obj_bboxes, gt_sub_bboxes, gt_obj_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            sub_bboxes ():
            obj_bboxes ():
            gt_sub_bboxes ():
            gt_obj_bboxes ():
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        gt_flags = sub_bboxes.new_zeros(sub_bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, sub_bboxes, obj_bboxes, gt_sub_bboxes, gt_obj_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
