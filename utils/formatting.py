# tensorboard へのロギングに役立つ関数群
from typing import Optional
import torch
import torchvision.utils as vutils


def visualize_posneg(images: torch.Tensor,
                     zero: Optional[torch.Tensor] = None,
                     positive: Optional[torch.Tensor] = None,
                     negative: Optional[torch.Tensor] = None):
    """Visualize 1-ch input images. By default, positive values are represented
    in black and negative ones are in red.

    Parameters
    ----------
    images : Tensor
        1-ch input images. size: [..., C(=1), H, W]
    zero : Optional[Tensor]
        Color to represent zeros. Default: [1., 1., 1.] (white)
    positive: Optional[Tensor]
        Color to represent positive values. Default: [0., 0., 0.] (black)
    negetive: Optional[Tensor]
        Color to represent negative values. Default: [1., 0., 0.] (red)

    Return
    ------
    out : Tensor
        3-ch images. size: [..., C, H, W]
    """
    device = images.device

    # Default arguments
    if zero is None:
        zero = torch.tensor([1., 1., 1.])
    zero = zero.view(3, 1, 1)
    zero = zero.to(device)

    if positive is None:
        positive = torch.tensor([0., 0., 0.])
    positive = positive.view(3, 1, 1)
    positive = positive.to(device)

    if negative is None:
        negative = torch.tensor([1., 0., 0.])
    negative = negative.view(3, 1, 1)
    negative = negative.to(device)

    pos = torch.clip(images, 0., 1.)
    pos = pos * positive + (1. - pos) * zero

    neg = - torch.clip(images, -1., 0.)
    neg = neg * negative + (1. - neg) * zero

    out = torch.where(images >= 0., pos, neg)
    return out


def interleave_tensors(ts: list[torch.Tensor], dim: int):
    """Interleave tensors."""


def interleave_images(imgs: list[torch.Tensor], pad_value: float = .5):
    """Interleave image tensors and tile them to make a single image.

    Parameter
    ---------
    imgs : list[torch.Tensor]
        All elements must have the same number of images and
        all images must have the same size.
    pad_value : float
        Default 0.5.

    Return
    ------
    tiled : torch.Tensor
        An image tensor.

    Example
    -------
    If inputs are x = [x1, x2, ..., xn], y = [y1, y2, ..., yn],
    and z = [z1, z2, ..., zn], then the output image looks like
        +----+----+----+
        | x1 | y1 | z1 |
        +----+----+----+
        | x2 | y2 | z2 |
        +----+----+----+
        ...
        +----+----+----+
        | xn | yn | zn |
        +----+----+----+
    """
    # make_grid() に渡すために [x1, y1, z1, x2, y2, z2, ...] という順番のリストを作る
    img_size = imgs[0].size()[1:]
    arranged_imgs = torch.stack(imgs, dim=1).view(-1, *img_size)
    # 一枚の画像にする
    img_grid = vutils.make_grid(arranged_imgs, nrow=len(imgs), pad_value=.5)
    return img_grid


def interleave_labels(labels: list[torch.Tensor]):
    # NOTE: interleave_image と似ている
    # ワンライナーだがマシな名前を付ける
    # [B, D] * n -> [B, n, D] -> [Bn, D] (=H, W)
    arranged = torch.stack(labels, dim=1).view(-1, labels[0].size(1))
    return arranged
