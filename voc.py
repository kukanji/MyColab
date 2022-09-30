import os.path as osp
import xml.etree.ElementTree as ElementTree
import numpy as np
import sys
import torch
import torch.utils.data as data
import cv2

ROOT_PATH = '/content/drive/MyDrive/MyColab/object_detection'
sys.path.append(ROOT_PATH)
from augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, \
        RandomMirror, ToPercentCoords, Resize, SubtractMeans


def make_filepath_list(rootpath):
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list



class GetBBoxAndLabel(object):

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):

        annotation = []

        xml = ElementTree.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            grid = ['xmin', 'ymin', 'xmax', 'ymax']

            for gr in (grid):
                axis_value = int(bbox.find(gr).text) - 1
                if gr == 'xmin' or gr == 'xmax':
                    axis_value /= width
                else:
                    axis_value /= height
                bndbox.append(axis_value)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            annotation += [bndbox]
        return np.array(annotation)


class DataTransform(object):

    def __init__(self, input_size, color_mean):

        self.transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
    
    def __call__(self, img, phase, boxes, labels):

        return self.transform[phase](img, boxes, labels) 


class PreprocessVOC2012(data.Dataset):

    def __init__(self, img_list, anno_list, phase, transform, get_bbox_label):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.get_bbox_label = get_bbox_label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        im, bl, _, _ = self.pull_item(index)
        return im, bl

    def pull_item(self, index):

        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        height, width, _  = img.shape

        anno_file_path = self.anno_list[index]
        bbox_label = self.get_bbox_label(anno_file_path, width, height)

        img, boxes, labels = self.transform(
            img,
            self.phase,
            bbox_label[:, :4],
            bbox_label[:, 4]
        )

        img = torch.from_numpy(
            img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        boxlbl = np.hstack(
            (boxes, np.expand_dims(labels, axis = 1))
        )
        return img, boxlbl, height, width

def multiobject_collate_fn(batch):

    imgs = []
    targets = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        
    imgs = torch.stack(imgs, dim = 0)

    return imgs, targets