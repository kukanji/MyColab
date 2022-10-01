import torch.nn as nn

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

    