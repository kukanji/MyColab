import torch.nn as nn
import torch
import torch.nn.init as init
from itertools import product as product
from math import sqrt as sqrt

def make_vgg():
    layers = []
    in_channels = 3
    cfg = [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 'MC',
           512, 512, 512, 'M',
           512, 512, 512
           ]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride = 2)]

        elif v == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride = 2, ceil_mode = True)]

        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    
    layers += [pool5,
                conv6, nn.ReLU(inplace=True),
                conv7, nn.ReLU(inplace=True)]
     
    return nn.ModuleList(layers)


def make_extras():
    layers = []
    in_channels = 1024
    cfg = [256, 512,
           128, 256,
           128, 256,
           128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size = (1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size = (3), stride = 2, padding = 1)]
    
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size = (1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size = (3), stride = 2, padding = 1)]

    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size = (1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size = (3))]
    
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size = (1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size = (3))]
    
    return nn.ModuleList(layers)


def make_loc(dbox_num = [4, 6, 6, 6, 4, 4]):
    
    loc_layers = []
    loc_layers += [nn.Conv2d(512, dbox_num[0] * 4, kernel_size=3, padding=1)]
    
    loc_layers += [nn.Conv2d(1024, dbox_num[1] * 4, kernel_size=3, padding=1)]
    
    loc_layers += [nn.Conv2d(512, dbox_num[2] * 4, kernel_size=3, padding=1)]
    
    loc_layers += [nn.Conv2d(256, dbox_num[3] * 4, kernel_size=3, padding=1)]
    
    loc_layers += [nn.Conv2d(256, dbox_num[4] * 4, kernel_size=3, padding=1)]
    
    loc_layers += [nn.Conv2d(256, dbox_num[5] * 4, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers)


def make_conf(classes_num = 21, dbox_num = [4, 6, 6, 6, 4, 4]):

    conf_layers = []

    conf_layers += [nn.Conv2d(512, dbox_num[0] * classes_num, kernel_size=3, padding=1)]
    
    conf_layers += [nn.Conv2d(1024, dbox_num[1] * classes_num, kernel_size=3, padding=1)]
    
    conf_layers += [nn.Conv2d(512, dbox_num[2] * classes_num, kernel_size=3, padding=1)]
    
    conf_layers += [nn.Conv2d(256, dbox_num[3] * classes_num, kernel_size=3, padding=1)]
    
    conf_layers += [nn.Conv2d(256, dbox_num[4] * classes_num, kernel_size=3, padding=1)]
    
    conf_layers += [nn.Conv2d(256, dbox_num[5] * classes_num, kernel_size=3, padding=1)]

    return nn.ModuleList(conf_layers)


class L2Norm(nn.Module):
    
    def __init__(self, input_channels = 512, scale = 20):

        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        
        init.constant_(self.weight, self.scale)

    def forward(self, x):

        norm = x.pow(2).sum(dim = 1, keepdim = True).sqrt() + self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


class DBox(object):

    def __init__(self, cfg):
        super(DBox, self).__init__()

        self.image_size = cfg['input_size']
        self.feature_maps= cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps'])
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']

    def make_dbox_list(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat = 2):
                f_k = self.image_size / self.steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max = 1, min = 0)

        return output
        