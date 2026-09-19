import torch


def span_xx_to_cxw(xx_spans):
    """
    Args:
        xx_spans: tensor, (#windows, 2) or (..., 2), each row is a window of format (st, ed)

    Returns:
        cxw_spans: tensor, (#windows, 2), each row is a window of format (center=(st+ed)/2, width=(ed-st))
    >>> spans = torch.Tensor([[0, 1], [0.2, 0.4]])
    >>> span_xx_to_cxw(spans)
    tensor([[0.5000, 1.0000],
        [0.3000, 0.2000]])
    >>> spans = torch.Tensor([[[0, 1], [0.2, 0.4]]])
    >>> span_xx_to_cxw(spans)
    tensor([[[0.5000, 1.0000],
         [0.3000, 0.2000]]])
    """
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)


def span_cxw_to_xx(cxw_spans):
    """
    Args:
        cxw_spans: tensor, (#windows, 2) or (..., 2), the last dim is a row denoting a window of format (center, width)

    >>> spans = torch.Tensor([[0.5000, 1.0000], [0.3000, 0.2000]])
    >>> span_cxw_to_xx(spans)
    tensor([[0.0000, 1.0000],
        [0.2000, 0.4000]])
    >>> spans = torch.Tensor([[[0.5000, 1.0000], [0.3000, 0.2000]]])
    >>> span_cxw_to_xx(spans)
    tensor([[[0.0000, 1.0000],
        [0.2000, 0.4000]]])
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)


def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union


def temporal_intersection_over_pred(gt_spans, pred_spans):
    """ intersection over the second input spans
    Args:
        gt_spans: (N, 2),
        pred_spans: (M, 2)

    Returns:

    """
    left = torch.max(gt_spans[:, None, 0], pred_spans[:, 0])
    right = torch.min(gt_spans[:, None, 1], pred_spans[:, 1])

    inter = (right - left).clamp(min=0)  # (N, M)
    inter_over_pred = inter / (pred_spans[:, 1] - pred_spans[:, 0])
    return inter_over_pred


def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area


def temporal_diou(spans1, spans2, eps=1e-7):
    """
    Distance-IoU for 1D temporal spans.
    Reference: https://arxiv.org/pdf/1911.08287.pdf

    Args:
        spans1: (N, 2) torch.Tensor, each row is [st, ed]
        spans2: (M, 2) torch.Tensor, each row is [st, ed]
        eps: A small value to avoid division by zero.

    Returns:
        diou: (N, M) torch.Tensor
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()

    # 1. Calculate standard IoU
    iou, _ = temporal_iou(spans1, spans2)

    # 2. Calculate centers of spans
    center1 = (spans1[:, 0] + spans1[:, 1]) / 2
    center2 = (spans2[:, 0] + spans2[:, 1]) / 2
    
    # 3. Calculate enclosing span
    left = torch.min(spans1[:, None, 0], spans2[:, 0])   # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    
    # Diagonal of the enclosing span (in 1D, this is just the width)
    c_diag = right - left
    
    # Squared distance between centers
    center_dist_sq = (center1[:, None] - center2).pow(2)

    # Penalty term
    penalty = center_dist_sq / c_diag.pow(2).clamp(min=eps)

    return iou - penalty


def temporal_ciou(spans1, spans2, eps=1e-7):
    """
    Complete-IoU for 1D temporal spans.
    Reference: https://arxiv.org/pdf/1911.08287.pdf

    Args:
        spans1: (N, 2) torch.Tensor, each row is [st, ed]
        spans2: (M, 2) torch.Tensor, each row is [st, ed]
        eps: A small value to avoid division by zero.

    Returns:
        ciou: (N, M) torch.Tensor
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()

    # 1. Calculate standard IoU
    iou, _ = temporal_iou(spans1, spans2)

    # 2. Calculate centers of spans
    center1 = (spans1[:, 0] + spans1[:, 1]) / 2
    center2 = (spans2[:, 0] + spans2[:, 1]) / 2
    
    # 3. Calculate enclosing span
    left = torch.min(spans1[:, None, 0], spans2[:, 0])   # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    
    # Diagonal of the enclosing span (in 1D, this is just the width)
    c_diag = right - left
    
    # Squared distance between centers
    center_dist_sq = (center1[:, None] - center2).pow(2)

    # DIoU penalty term
    diou_penalty = center_dist_sq / c_diag.pow(2).clamp(min=eps)

    # 4. Add width consistency penalty
    w1 = spans1[:, 1] - spans1[:, 0]
    w2 = spans2[:, 1] - spans2[:, 0]
    # The arctan is used to normalize the width difference to the range [0, 1]
    v = (4 / (torch.pi ** 2)) * (torch.atan(w1[:, None] / w2.clamp(min=eps)) - torch.atan(w2 / w1[:, None].clamp(min=eps))).pow(2)
    
    # Alpha is a trade-off parameter
    # The `detach()` is important: we don't want to backpropagate through alpha.
    alpha = v / (1 - iou + v).clamp(min=eps)
    
    ciou_penalty = diou_penalty + alpha.detach() * v

    return iou - ciou_penalty


if __name__ == "__main__":
    span1 = torch.Tensor([[0, 1], [2, 6]])
    span2 = torch.Tensor([[0, 1], [2, 3]])

    print(f"temporal_iou: {temporal_iou(span1, span2)[0]}")
    print(f"generalized_temporal_iou: {generalized_temporal_iou(span1, span2)}")
    print(f"temporal_diou: {temporal_diou(span1, span2)}")
    print(f"temporal_ciou: {temporal_ciou(span1, span2)}")
