# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


@TASK_UTILS.register_module()
class BboxOverlaps2D_GLIP(BboxOverlaps2D):

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        TO_REMOVE = 1
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + TO_REMOVE) * (
            bboxes1[:, 3] - bboxes1[:, 1] + TO_REMOVE)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + TO_REMOVE) * (
            bboxes2[:, 3] - bboxes2[:, 1] + TO_REMOVE)

        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [N,M,2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou


@TASK_UTILS.register_module()
class BboxDistanceMetric(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""
    def __init__(self, constant=12.8):
        self.constant = constant

    def __call__(self, bboxes1, bboxes2, mode='wasserstein', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned, constant=self.constant)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6, constant=12.8):
    assert mode in ['iou', 'iof', 'giou', 'normalized_giou', 'ciou', 'diou', 'wasserstein', 'kl'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = torch.max(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap + eps

    if mode in ['giou', 'normalized_giou', 'ciou', 'diou']:
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])
        

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    
    if mode in ['iou', 'iof']:
        return ious
    
    # calculate gious
    if mode in ['giou', 'normalized_giou', 'ciou', 'diou']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

    if mode == 'giou':
        return gious

    if mode == 'kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(1+kl)

        return kld

    if mode == 'normalized_giou':
        gious = (1 + gious) / 2
        
        return gious

    if mode == 'diou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #distances of center points between gt and pre

        enclosed_diagonal_distances = enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1] # distances of diagonal of enclosed bbox
        
        dious = ious - center_distance / torch.max(enclosed_diagonal_distances, eps)
        
        dious = torch.clamp(dious,min=-1.0,max = 1.0)
        
        return dious

    if mode == 'ciou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps # distances of center points between gt and pre

        enclosed_diagonal_distances = enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1] # distances of diagonal of enclosed bbox

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        cious = ious - (center_distance / torch.max(enclosed_diagonal_distances, eps) + v ** 2 / torch.max(1 - ious + v, eps))

        cious = torch.clamp(cious, min=-1.0, max=1.0)
        
        return cious
    
    if mode == 'wasserstein':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wassersteins = torch.sqrt(center_distance + wh_distance)

        normalized_wasserstein = torch.exp(-wassersteins/constant)

        return normalized_wasserstein
