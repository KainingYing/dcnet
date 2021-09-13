# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns hois to ground truth hois."""

    @abstractmethod
    def assign(self,
               sub_bboxes,
               obj_bboxes,
               obj_labels,
               verb_labels,
               gt_sub_bboxes,
               gt_obj_bboxes,
               gt_obj_labels,
               gt_verb_labels,
               **kwargs):
        """Assign hois to either a ground truth hois or a negative hois."""
