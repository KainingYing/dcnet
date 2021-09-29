import cv2
import torch
import numpy as np

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val


imshow_backend = 'cv2'
supported_backends = ['cv2', 'plt']

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def imshow(img, win_name='', wait_time=0, backend='plt'):
    """Show an image.

    Args:
        img (str or ndarray or tensor): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        backend (str): cv2 or plt (matplotlib)
    """
    if backend is None:
        backend = imshow_backend
    if backend not in supported_backends:
        raise ValueError(f'backend: {backend} is not supported for imshow.'
                         f"Supported backends are {imshow_backend}")
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy() if img.is_cuda else img.numpy()
    if backend == 'cv2':
        cv2.imshow(win_name, imread(img))
        if wait_time == 0:  # prevent from hanging if windows was closed
            while True:
                ret = cv2.waitKey(1)

                closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                # if user closed window or if some key pressed
                if closed or ret != -1:
                    break
        else:
            ret = cv2.waitKey(wait_time)
    else:  # plt
        plt.imshow(imread(img))
        plt.show()


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.
    # note: the bboxes shaped with (k, 4)

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy() if img.is_cuda else img.numpy()
    img = imread(img)
    img = np.ascontiguousarray(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img


if __name__ == '__main__':
    import torch
    import zjutcv
    import numpy as np
    img = torch.zeros(300, 300, 3)
    bboxes = np.array([[0, 0, 100, 100]])
    # zjutcv.imshow(img, backend='plt')
    imshow_bboxes(img, bboxes)



