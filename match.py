import torch
# dboxの情報をbbox形式に変換する関数
def point_form(boxes):

    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2, 1))

    