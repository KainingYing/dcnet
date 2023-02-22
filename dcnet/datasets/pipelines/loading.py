# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import HOI_PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@HOI_PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmhoidet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@HOI_PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@HOI_PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_sub_bbox=True,
                 with_obj_bbox=True,
                 with_obj_label=True,
                 with_verb_label=True,
                 file_client_args=dict(backend='disk'),
                 to_float32=True):
        self.with_sub_bbox = with_sub_bbox
        self.with_obj_bbox = with_obj_bbox
        self.with_obj_label = with_obj_label
        self.with_verb_label = with_verb_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.to_float32 = to_float32

    def _load_sub_bboxes(self, results):
        """Private function to load subject (person) bounding box annotations."""

        ann_info = results['ann_info']
        if self.to_float32:
            results['gt_sub_bboxes'] = ann_info['sub_bboxes'].astype(np.float32)  # astype will create a new id, no need copy()
        else:
            results['gt_sub_bboxes'] = ann_info['sub_bboxes']
        results['bbox_fields'].append('gt_sub_bboxes')
        return results

    def _load_obj_bboxes(self, results):
        """Private function to load object bounding box annotations."""

        ann_info = results['ann_info']
        if self.to_float32:
            results['gt_obj_bboxes'] = ann_info['obj_bboxes'].astype(np.float32)
        else:
            results['gt_obj_bboxes'] = ann_info['obj_bboxes']
        results['bbox_fields'].append('gt_obj_bboxes')
        return results

    def _load_obj_labels(self, results):
        """Private function to load object label annotations."""

        results['gt_obj_labels'] = results['ann_info']['obj_labels'].copy()
        results['label_fields'].append('gt_obj_labels')
        return results

    def _load_verb_labels(self, results):
        """Private function to load verb label annotations."""

        results['gt_verb_labels'] = results['ann_info']['verb_labels'].copy()
        results['label_fields'].append('gt_verb_labels')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmhoidet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        assert self.with_sub_bbox and self.with_obj_bbox and self.with_obj_label and \
               self.with_verb_label, 'Must contain four type annotations at the same time'

        if self.with_sub_bbox:
            results = self._load_sub_bboxes(results)
            if results is None:
                return None
        if self.with_obj_bbox:
            results = self._load_obj_bboxes(results)
        if self.with_obj_label:
            results = self._load_obj_labels(results)
        if self.with_verb_label:
            results = self._load_verb_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_sub_bbox={self.with_sub_bbox}, '
        repr_str += f'(with_obj_bbox={self.with_obj_bbox}, '
        repr_str += f'with_obj_label={self.with_obj_label}, '
        repr_str += f'with_verb_label={self.with_verb__label}, '
        return repr_str


@HOI_PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, min_gt_bbox_wh):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results
