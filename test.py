import torch
import torch.nn.functional as F

# from mmhoidet.models.losses.utils import weight_reduce_loss
from mmhoidet.models.losses.focal_loss import ElementWiseFocalLoss, py_sigmoid_focal_loss

def main():
   fl = ElementWiseFocalLoss(use_sigmoid=True)
   pred = [
       [-100, 100, 0],
       [0, 100, 100]
   ]
   target = [
       [0, 1, 0],
       [0, 1, 1]
   ]
   pred = torch.tensor(pred, dtype=torch.float)
   target = torch.tensor(target, dtype=torch.float)
   print(fl(pred, target))
   print(py_sigmoid_focal_loss(pred, target))


if __name__ == '__main__':
    # pred = torch.randn(100, 117)
    # target = torch.randn(100, 117)
    # pred = [
    #     [-100, 100, 0],
    #     [0, 100, 100]
    # ]
    # target = [
    #     [0, 1, 0],
    #     [0, 1, 1]
    # ]
    # pred = torch.tensor(pred, dtype=torch.float)
    # target = torch.tensor(target, dtype=torch.float)
    # weight = torch.tensor([0, 1])
    # print(py_sigmoid_focal_loss(pred, target, weight))

    main()