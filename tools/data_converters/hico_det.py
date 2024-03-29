import argparse
import os.path as osp
from pathlib import Path

import mmcv

ANNO_FILE = {
    'train': 'annotations/trainval_hico.json',
    'test': 'annotations/test_hico.json'
}

IMG_PREFIX = {
    'train': 'images/train2015/',
    'test': 'images/test2015/'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add images' shape into HICO_DET annotations"
    )
    parser.add_argument('--data_root', default='./data/hico_20160224_det/', help='Dataset root')
    args = parser.parse_args()
    return args


def add_shape2annotations(data_root, data_type):
    anno_file = osp.join(data_root, ANNO_FILE[data_type])
    img_prefix = osp.join(data_root, IMG_PREFIX[data_type])
    data_infos = mmcv.load(anno_file)

    progress_bar = mmcv.ProgressBar(len(data_infos))

    for i, item in enumerate(data_infos):
        filename = osp.join(img_prefix, Path(item['file_name']).name)
        img = mmcv.imread(filename)
        width, height = img.shape[:2]
        item['width'], item['height'] = width, height

        progress_bar.update()

    # TODO: overwrite the origin anno_file
    # TODO: 使用懒读取 pillow的方法，可不可以变快一点
    mmcv.dump(data_infos, anno_file)


def main():
    args = parse_args()
    assert 'hico' in args.data_root, 'This hint that we only consider the HICODet,' \
                                     'So better rename the dataset folder with key name hico'
    for data_type in ['train', 'test']:
        add_shape2annotations(args.data_root, data_type)
        print(f"{data_type} is finished!")


if __name__ == '__main__':
    main()
