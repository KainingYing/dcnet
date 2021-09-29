# Copyright (c) OpenMMLab. All rights reserved.
import mmcv


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def hico_det_classes():
    """object classes and verb classes"""
    return (['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
             'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
            ['adjust', 'assemble', 'block', 'blow', 'board', 'break',
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
             'watch', 'wave', 'wear', 'wield', 'zip']
            )


dataset_aliases = {
    'hico_det': ['hicodet', 'hico-det', 'hico', 'hico_det']
}


# TODO: must be modified
def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
