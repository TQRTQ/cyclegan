import torch.nn as nn
import torch

##############################
#        resnet-hess
##############################
class ResBlock_hess(nn.Module):

    def __init__(self, in_channels: int, apply_dropout: bool = True):
        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        super().__init__()

        conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        layers = [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dropout:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)

##############################
#        Generator-hess
##############################
class Generator_hess(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 apply_dropout: bool = True,
                 image_size: int = 128
                 ):

        super().__init__()

        f = 1
        num_downsampling = 2
        num_resnet_blocks = 6 if image_size == 128 else 9
        # num_resnet_blocks = 9 if image_size == 128 else 9


        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size=3, stride=2, padding=1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resnet_blocks):
            resnet_block = ResBlock_hess(in_channels=out_channels * f, apply_dropout=apply_dropout)
            self.layers += [resnet_block]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f // 2), 3, 2, padding=1, output_padding=1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f // 2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=7, stride=1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.ReLU()]
        # self.layers += [nn.ReflectionPad2d(3), conv, nn.()]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):

        x = self.net(x)

        return x



class constrain(nn.Module):
    def __init__(self, min_value, max_value):
        super(constrain, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=self.min_value, max=self.max_value)



##################################
#     Generator-hess-constrain
##################################

class Generator_hess_constrain(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 apply_dropout: bool = True,
                 image_size: int = 128
                 ):

        super().__init__()

        f = 1
        num_downsampling = 2
        num_resnet_blocks = 6 if image_size == 128 else 9
        # num_resnet_blocks = 9 if image_size == 128 else 9


        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size=3, stride=2, padding=1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resnet_blocks):
            resnet_block = ResBlock_hess(in_channels=out_channels * f, apply_dropout=apply_dropout)
            self.layers += [resnet_block]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f // 2), 3, 2, padding=1, output_padding=1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f // 2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=7, stride=1)
        # self.layers += [nn.ReflectionPad2d(3), conv, nn.ReLU()]
        self.layers += [nn.ReflectionPad2d(3), conv]

        self.net = nn.Sequential(*self.layers)
        self.constrain = constrain(min_value=0, max_value=100)

    def forward(self, x):

        x = self.net(x)
        x = self.constrain(x)

        return x












##############################
#        Discriminator-hess
##############################
class Discriminator_hess(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 num_layers: int = 3):

        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """

        super().__init__()
        in_f  = 1
        out_f = 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, num_layers):
            conv = nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f   = out_f
            out_f *= 2

        out_f = min(2 ** num_layers, 8)
        conv = nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): return self.net(x)





##############################
#        Initializer-hess
##############################
class Initializer_hess:

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """

        self.init_type = init_type
        self.init_gain = init_gain

    def init_module(self, m):

        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val=0);

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)
            nn.init.constant_(m.bias.data, val=0)

    def __call__(self, net):

        """
        Parameters:
            net: Network
        """

        net.apply(self.init_module)

        return net