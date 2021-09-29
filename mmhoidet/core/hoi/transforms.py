import torch
import numpy as np


def hoi2result(instance_labels, verb_scores, bboxes, sub_ids, obj_ids, max_per_img=None, valid_hois=None):
    """Convert detection hois to a list of numpy arrays.
    Used in QPIC.

    Args:
        valid_hois ():
        max_per_img ():
        obj_ids ():
        sub_ids ():
        verb_scores ():
        instance_labels ():
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)

    Returns:
        list(ndarray): bbox results of each class
    """
    if isinstance(bboxes, torch.Tensor):
        instance_labels = instance_labels.detach().cpu().numpy()
        verb_scores = verb_scores.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        sub_ids = sub_ids.detach().cpu().numpy()
        obj_ids = obj_ids.detach().cpu().numpy()

    bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(bboxes, instance_labels)]

    hoi_scores = verb_scores
    verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
    sub_ids = np.tile(sub_ids, (hoi_scores.shape[1], 1)).T
    obj_ids = np.tile(obj_ids, (hoi_scores.shape[1], 1)).T

    hoi_scores = hoi_scores.ravel()
    verb_labels = verb_labels.ravel()
    sub_ids = sub_ids.ravel()
    obj_ids = obj_ids.ravel()

    if len(sub_ids) > 0:
        obj_labels = np.array([bboxes[obj_id]['category_id'] for obj_id in obj_ids])
        masks = valid_hois[verb_labels, obj_labels]
        hoi_scores *= masks
        # note: the label of verb is 1-based.
        hois = [{'subject_id': sub_id, 'object_id': obj_id, 'category_id': category_id + 1, 'score': score} for
                sub_id, obj_id, category_id, score in zip(sub_ids, obj_ids, verb_labels, hoi_scores)]
        hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        # todo: consider put this in head in the future
        if max_per_img:
            hois = hois[:max_per_img]
    else:
        hois = []
    return {'predictions': bboxes, 'hoi_prediction': hois}  # Exactly corresponds to annotations and hoi_prediction
