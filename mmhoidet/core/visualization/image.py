# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_interactions(img,
                            sub_bboxes,
                            obj_bboxes,
                            obj_labels,
                            verb_labels,
                            obj_names=None,
                            verb_names=None,
                            score_thr=0,
                            subject_color='blue',
                            object_color='yellow',
                            interaction_color='green',
                            thickness=2,
                            font_size=13,
                            win_name='',
                            show=True,
                            wait_time=0,
                            out_file=None):
    """Draw interactions and class labels (with scores) on an image.
    This function is only used to visualize the GroundTruth annotations now.

    Args:
        img (str or ndarray): The image to be displayed.
        sub_bboxes (ndarray): Subject boxes (person), shaped (n, 4)
        obj_bboxes (ndarray): Object boxes (COCO), shaped (n, 4)
        obj_labels (ndarray): Labels of object boxes, shaped (n, )
        verb_labels: (ndarray): One-hot label of verb of each interactions, shaped (n, 117)
        obj_names:
        verb_names:
        subject_color:
        object_color:
        interaction_color:
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """

    assert sub_bboxes.ndim == 2, \
        f' sub_bboxes ndim should be 2, but its ndim is {sub_bboxes.ndim}.'
    assert obj_bboxes.ndim == 2, \
        f' obj_bboxes ndim should be 2, but its ndim is {obj_bboxes.ndim}.'
    assert obj_labels.ndim == 1, \
        f' obj_labels ndim should be 1, but its ndim is {obj_labels.ndim}.'
    assert verb_labels.ndim == 2, \
        f' verb_labels ndim should be 1, but its ndim is {verb_labels.ndim}.'
    assert sub_bboxes.shape[0] == obj_bboxes.shape[0] == obj_labels.shape[0] == \
           verb_labels.shape[0], f'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert sub_bboxes.shape[1] == 4, f' sub_bboxes.shape[1] should be 4 here'

    img = mmcv.imread(img).astype(np.uint8)

    # set the visualize color
    sub_color = color_val_matplotlib(subject_color)
    obj_color = color_val_matplotlib(object_color)
    interaction_color = color_val_matplotlib(interaction_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    sub_polygons = []
    obj_polygons = []
    sub_colors = []
    obj_colors = []
    for i, (sub_bbox, obj_bbox, obj_label, verb_label) in enumerate(
            zip(sub_bboxes, obj_bboxes, obj_labels, verb_labels)):
        sub_bbox_int = sub_bbox.astype(np.int32)
        obj_bbox_int = obj_bbox.astype(np.int32)
        sub_poly = [[sub_bbox_int[0], sub_bbox_int[1]], [sub_bbox_int[0], sub_bbox_int[3]],
                    [sub_bbox_int[2], sub_bbox_int[3]], [sub_bbox_int[2], sub_bbox_int[1]]]
        obj_poly = [[obj_bbox_int[0], obj_bbox_int[1]], [obj_bbox_int[0], obj_bbox_int[3]],
                    [obj_bbox_int[2], obj_bbox_int[3]], [obj_bbox_int[2], obj_bbox_int[1]]]
        sub_np_poly = np.array(sub_poly).reshape((4, 2))
        obj_np_poly = np.array(obj_poly).reshape((4, 2))
        sub_polygons.append(Polygon(sub_np_poly))
        obj_polygons.append(Polygon(obj_np_poly))
        sub_colors.append(sub_color)
        obj_colors.append(obj_color)

        # Hoi detection only considers the situation where subject is person.
        sub_label_text = 'person'
        obj_label_text = obj_names[
            obj_label] if obj_names is not None else f'class {obj_label}'

        # if len(sub_bbox) > 4:
        #     label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            sub_bbox_int[0],
            sub_bbox_int[1],
            f'{sub_label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=sub_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        ax.text(
            obj_bbox_int[0],
            obj_bbox_int[1],
            f'{obj_label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=obj_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')

        # visualize the interactions here.
        # inteaction label names
        inter_label = np.where(verb_label > 0)[0]
        inter_label_text = ('/').join(list(map(lambda x: verb_names[x], inter_label)))
        sub_center_x, sub_center_y = (sub_bbox_int[0] + sub_bbox_int[2]) / 2, (sub_bbox_int[1] + sub_bbox_int[3]) / 2
        obj_center_x, obj_center_y = (obj_bbox_int[0] + obj_bbox_int[2]) / 2, (obj_bbox_int[1] + obj_bbox_int[3]) / 2
        # step 1:visualize the line
        ax.plot([sub_center_x, obj_center_x], [sub_center_y, obj_center_y], color=interaction_color, )

        # step 2:visualize the category name
        ax.text(
            (sub_center_x + obj_center_x) / 2,
            (sub_center_y + obj_center_y) / 2,
            f'{inter_label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=interaction_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        # step 3:visualize the point
        ax.plot(sub_center_x, sub_center_y,
                color=sub_color,
                marker='o',
                )
        ax.plot(obj_center_x, obj_center_y,
                color=obj_color,
                marker='o'
                )

    plt.imshow(img)

    sub_p = PatchCollection(
        sub_polygons, facecolor='none', edgecolors=sub_colors, linewidths=thickness)
    obj_p = PatchCollection(
        obj_polygons, facecolor='none', edgecolors=obj_colors, linewidths=thickness)

    ax.add_collection(sub_p)
    ax.add_collection(obj_p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img
