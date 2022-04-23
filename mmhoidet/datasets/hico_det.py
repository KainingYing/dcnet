import time
import warnings
import os
from pathlib import Path
from collections import defaultdict

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .builder import HOI_DATASETS
from .pipelines import Compose


@HOI_DATASETS.register_module()
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

    VERB_CLASSES = ('adjust', 'assemble', 'block', 'blow', 'board', 'break',
                    'brush_with', 'buy', 'carry', 'catch', 'chase', 'check',
                    'clean', 'control', 'cook', 'cut', 'cut_with', 'direct',
                    'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at',
                    'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind',
                    'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose',
                    'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss',
                    'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light',
                    'load', 'lose', 'make', 'milk', 'move', 'no_interaction',
                    'open', 'operate', 'pack', 'paint', 'park', 'pay',
                    'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push',
                    'race', 'read', 'release', 'repair', 'ride',
                    'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign',
                    'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze',
                    'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag',
                    'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash',
                    'watch', 'wave', 'wear', 'wield', 'zip')

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix='',
                 test_mode=False,
                 obj_classes=None,
                 verb_classes=None,
                 valid_hois_file=None,
                 mode='train'):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.mode = mode
        self.valid_hois_file = valid_hois_file
        # Get the object and category classes names in order.
        self.OBJ_CLASSES = self.get_obj_classes(obj_classes)
        self.VERB_CLASSES = self.get_verb_classes(verb_classes)
        self.valid_hois = self._get_valid_hois()

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

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

        assert self.mode in ['train', 'val', 'test'], 'Dataset mode must be ' \
                                                      'one of the ["train", "val", "test"]'
        if mode in ['val', 'test']:
            self._set_rare_hois()

    def _set_rare_hois(self):
        counts = defaultdict(lambda: 0)

        for img_anno in self.data_infos:
            hois = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (self._valid_obj_cat_ids.index(bboxes[hoi['subject_id']]['category_id']),
                           self._valid_obj_cat_ids.index(bboxes[hoi['object_id']]['category_id']),
                           self._valid_verb_cat_ids.index(hoi['category_id']))
                counts[triplet] += 1
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        print('loading annotations into memory...')
        tic = time.time()
        dataset = mmcv.load(ann_file)
        assert type(dataset) == list, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        return dataset

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
            verb_vec = np.zeros(len(self.VERB_CLASSES), dtype=np.float32)  # this data type must align the preds
            verb_vec[self.verb_cat2label[verb_label]] = 1.0
            verb_labels.append(verb_vec)
        return dict(sub_bboxes=np.array(sub_bboxes), obj_bboxes=np.array(obj_bboxes), obj_labels=np.array(obj_labels),
                    verb_labels=np.array(verb_labels))

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []  # gt_sub_bboxes & gt_obj_bboxes
        results['label_fields'] = []  # gt_obj_bboxes & gt_verb_bboxes

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
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get data from pipeline.

        Args:
            idx (int): Index of data.

        Returns:
        """
        img_info = self.data_infos[idx]
        img_info.setdefault('filename', img_info['file_name'])
        results = dict(img_info=img_info)
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
            if data is None:  # re-sample a data if the data is None
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

    def _get_valid_hois(self):
        """get valid hois from disk"""
        if self.valid_hois_file is None:
            warnings.warn('Valid_hois_file is not specified, so use the `.../annotations/corre_hico.npy` instead.')
            self.valid_hois_file = Path(self.ann_file).parent / "corre_hico.npy"
        else:
            assert Path(self.valid_hois_file).suffix == '.npy', f'The valid_hois_file must has file with ".npy", ' \
                                                                f'but get type "{Path(self.valid_hois_file).suffix}"'
        return np.load(self.valid_hois_file)

    def _filter_imgs(self, min_size=32):
        """Filter images too small and without ground truths"""
        valid_inds = []
        assert 'width' in self.data_infos[0].keys() and 'height' in self.data_infos[0].keys(), 'Please run the script ' \
                                                                                               '`./tools/mics/dataset_processing.py` first.'

        for i, img_info in enumerate(self.data_infos):
            # TODO:consider the empty GT case here
            if img_info['width'] >= min_size and img_info['height'] >= min_size:
                valid_inds.append(i)

        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None):
        """
        1. Get ground truth
        2. self.sum_gts and self.gt_triplets
        Args:
            results:
            metric:
            logger:

        Returns:

        """
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.gt_triplets = []
        # self.gts = []
        self.sum_gts = defaultdict(lambda: 0)
        for gts in self.data_infos:
            for bbox in gts['annotations']:
                bbox['category_id'] = self.obj_cat2label[bbox['category_id']]
            for hoi in gts['hoi_annotation']:
                triplet = (gts['annotations'][hoi['subject_id']]['category_id'],
                           gts['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])  # (sub_cate, obj_cate, hoi_cate)
                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1

        for img_preds, img_gts in zip(results, self.data_infos):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare,
                                                                                   m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov = iou_mat.copy()
        iou_mat[iou_mat >= self.overlap_iou] = 1
        iou_mat[iou_mat < self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i], pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
            S_rec2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line + 1) * (bottom_line - top_line + 1)
                return intersect / (sum_area - intersect)
        else:
            return 0
