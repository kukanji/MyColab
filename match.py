import torch
# dboxの情報をbbox形式に変換する関数
def point_form(boxes):

    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2, 1))

#2個のボックスが重なる面積を求める関数
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2)
    )

    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )

    inter = torch.clamp((max_xy - min_xy), min = 0)

    return inter[:, :, 0] * inter[:, :, 1]

#jaccard係数(IoU)を求める関数
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union
