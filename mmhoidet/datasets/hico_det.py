import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class HICODet(Dataset):
    OBJ_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                   'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    VERB_CLASSES = ('adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 'carry', 'catch', 'chase', 'check', 'clean',
                    'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at',
                    'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose',
                    'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light',
                    'load', 'lose', 'make', 'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
                    'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release', 'repair', 'ride',
                    'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit_at', 'sit_on', 'slide', 'smell',
                    'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag',
                    'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash', 'watch',
                    'wave', 'wear', 'wield', 'zip')

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix='',
                 test_mode=False,
                 obj_classes=None,
                 verb_classes=None):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        # Get the object and category classes names in order.
        self.OBJ_CLASSES = self.get_obj_classes(obj_classes)
        self.VERB_CLASSES = self.get_verb_classes(verb_classes)

        # Load data information and annotations.
        self.data_infos = self.load_annotations(self.ann_file)

        self._valid_obj_cat_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)  # object ids (identical to MS COCO)
        self.obj_cat2label = {ids: i for i, ids in enumerate(self._valid_obj_cat_ids)}
        self._valid_verb_cat_ids = list(range(1, 118))  # 117 action classes
        self.verb_cat2label = {ids: i for i, ids in enumerate(self._valid_verb_cat_ids)}

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmcv.load(ann_file)

    def get_ann_info(self, idx):
        """Get HICO DET annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        # img_id = self.data_infos[idx]['id']
        ann_info = self.data_infos[idx]
        sub_bboxes, obj_bboxes, obj_labels, verb_labels = [], [], [], []

        # a list used to mark used pair
        pair2idx = {}

        ins_annos = ann_info['annotations']
        for hoi_item in ann_info['hoi_annotation']:
            sub_id = hoi_item['subject_id']
            obj_id = hoi_item['object_id']
            verb_label = hoi_item['category_id']
            if (sub_id, obj_id) in pair2idx.keys():
                # note: only add a label to the verb label vector
                pair_id = pair2idx[(sub_id, obj_id)]
                verb_labels[pair_id][self.verb_cat2label[verb_label]] = 1
                continue

            sub_bbox = ins_annos[sub_id]['bbox']
            obj_bbox = ins_annos[obj_id]['bbox']
            obj_label = ins_annos[obj_id]['category_id']
            pair2idx[(sub_id, obj_id)] = len(pair2idx)  # mark this pair exists

            sub_bboxes.append(sub_bbox)
            obj_bboxes.append(obj_bbox)
            obj_labels.append(self.obj_cat2label[obj_label])
            verb_vec = np.zeros(len(self.VERB_CLASSES))
            verb_vec[self.verb_cat2label[verb_label]] = 1
            verb_labels.append(verb_vec)

        return dict(sub_bboxes=np.array(sub_bboxes), obj_bboxes=np.array(obj_bboxes), obj_labels=np.array(obj_labels), verb_labels=np.array(verb_labels))

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        # results['seg_prefix'] = self.seg_prefix
        # results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        # results['mask_fields'] = []
        # results['seg_fields'] = []

    def prepare_train_img(self, idx):
        """
        Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]  # including information and annotations
        img_info.setdefault('filename', img_info['file_name'])
        # TODO: how to construct the annotation format
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    @classmethod
    def get_obj_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.OBJ_CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    @classmethod
    def get_verb_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.VERB_CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names
