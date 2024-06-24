import math

import paddle
import paddle.nn as nn
from typing import Optional,Sequence
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
import paddle.nn.functional as F
from paddle.nn.initializer import Assign
from paddle.nn import Conv2D
from paddleseg.models.backbones.transformer_utils import (DropPath, ones_,
                                                          to_2tuple, zeros_)
from paddleseg.models import layers


class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias_attr=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(out_planes) if bn else None
        self.relu = nn.PReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    



class SWA(nn.Layer):
    def __init__(self, in_channel, out_channel=None, x=256):
        super(SWA,self).__init__()
        
        if out_channel is None:
            out_channel = in_channel
        self.pool_h_max = nn.AdaptiveMaxPool2D((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2D((1, None))
        
        self.max_h = nn.Sequential(
            nn.Conv1D(in_channel,in_channel,kernel_size=5,padding=2,groups=in_channel),
            nn.Conv1D(in_channel,out_channel,kernel_size=1,padding=0),
            nn.BatchNorm1D(in_channel)
        )
        self.max_w = nn.Sequential(
            nn.Conv1D(in_channel,in_channel,kernel_size=5,padding=2,groups=in_channel),
            nn.Conv1D(in_channel,out_channel,kernel_size=1,padding=0),
            nn.BatchNorm1D(in_channel)
        )
    def forward(self, x):
        x_h = self.pool_h_max(x).squeeze(axis=3)  # [1, 32, 100, 1]-->  [1, 32, 100]
        x_w = self.pool_w_max(x).squeeze(axis=2)
                     
        out_h10 = self.max_h(x_h)
        out_w10 = self.max_w(x_w)
        a_w = paddle.nn.functional.sigmoid(out_w10).unsqueeze(axis=2) # B N 1 W
        a_h = paddle.nn.functional.sigmoid(out_h10).unsqueeze(axis=3) # B N H 1
        out = x*a_w*a_h
        return out  
    

    
class Mlp(nn.Layer):
    """Multilayer perceptron."""
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.dwconv = get_depthwise_conv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class CSModule(nn.Layer):
    def __init__(self, in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5,7),
            dilations: Sequence[int] = (1, 1,1),
            expansion: float = 1.0,
            ):
        super(CSModule,self).__init__()
        hidden_channels = (int)(out_channels*expansion)
        
        
        self.pre_conv = BasicConv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       bn=True,relu=True)
        
        # #split 
        self.dw_conv = BasicConv(hidden_channels//2, hidden_channels//2, kernel_sizes[0], 1, kernel_sizes[0]//2,
                       dilations[0],groups=hidden_channels//2,bn=True,relu=None)
        
        self.h_conv1 = BasicConv(hidden_channels//4,hidden_channels//4,(1,5),1,(0,5//2),bn=None,relu=None,groups=hidden_channels//4)
        self.v_conv1 = BasicConv(hidden_channels//4,hidden_channels//4,(5,1),1,(5//2,0),bn=True,relu=None,groups=hidden_channels//4)
        
        self.h_conv2 = BasicConv(hidden_channels//4,hidden_channels//4,(1,7),1,(0,7//2),bn=None,relu=None,groups=hidden_channels//4)
        self.v_conv2 = BasicConv(hidden_channels//4,hidden_channels//4,(7,1),1,(7//2,0),bn=True,relu=None,groups=hidden_channels//4)
        
        self.pw_conv = BasicConv(hidden_channels, hidden_channels, 1, 1,0,
                       bn=True,relu=True)  # 
        
        self.swa_att = SWA(in_channels,in_channels)  #

        self.post_conv = BasicConv(hidden_channels,out_channels, 1, 1,0,
                       bn=True,relu=True)   #

    def forward(self, x):
        x = self.pre_conv(x)
        y = x
        _, c, _, _ = x.shape
        # 
        con_out0,con_out1,con_out2 = paddle.split(x, num_or_sections=[c//2,c//4,c//4], axis=1)
        
        # 3x3dw  +  1x5 5x1conv  +  1x7 7x1conv 
        x = paddle.concat([self.dw_conv(con_out0), self.v_conv1(self.h_conv1(con_out1)), self.v_conv2(self.h_conv2(con_out2))], axis=1)

        x = self.pw_conv(x)  #
        att_out = self.swa_att(y)

        x = x * att_out
        x = self.post_conv(x)
        return x



class DWConv(nn.Layer):
    def __init__(self, in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (5, 5,5),
            dilations: Sequence[int] = (1, 1,1),
            expansion: float = 1.0,
            ):
        super(DWConv,self).__init__()
        hidden_channels = (int)(out_channels*expansion)
        
        self.dw_conv = BasicConv(hidden_channels, hidden_channels, kernel_sizes[0], 1, kernel_sizes[0]//2,
                       dilations[0],groups=hidden_channels,bn=None,relu=None)

        self.pw_conv = BasicConv(hidden_channels, hidden_channels, 1, 1,0,
                       bn=True,relu=True)  #  

    def forward(self, x):
        shortcut = x
        # x = self.pre_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)  # 
        return x+shortcut
    

class CSModule_Upsampling(nn.Layer):
    def __init__(self, in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (5, 5,5),
            dilations: Sequence[int] = (1, 1,1),
            expansion: float = 1.0,
            ):
        super(CSModule_Upsampling,self).__init__()
        hidden_channels = (int)(out_channels*expansion)
        
        
        self.pre_conv = BasicConv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       bn=True,relu=True)
        
        self.dw_conv = BasicConv(hidden_channels, hidden_channels, kernel_sizes[0], 1, kernel_sizes[0]//2,
                       dilations[0],groups=hidden_channels,bn=None,relu=None)

        self.pw_conv = BasicConv(hidden_channels, hidden_channels, 1, 1,0,
                       bn=True,relu=True)  #  
        self.swa_att = SWA(in_channels,in_channels) #

        self.post_conv = BasicConv(hidden_channels,out_channels, kernel_size=1, stride=1, padding=0,
                       bn=True,relu=True)   #

    def forward(self, x):
        shortcut = x
        x = self.pre_conv(x)
        y = x

        x = self.dw_conv(x)
        x = self.pw_conv(x)  # 
        att_out = self.swa_att(y)
        x = x * att_out
        return x+shortcut
    

class CSBlock(nn.Layer):
    def __init__(self, 
                hidden_channels:int=256,
                kernel_sizes: Sequence[int] = (3, 5, 7),
                dilations: Sequence[int] = (1, 1, 1, 1, 1),
                ffn_scale: float = 2.0,
                dropout_rate: float = 0.,
                drop_path_rate: float = 0.0,
                ):
        super(CSBlock,self).__init__()

        self.norm1 = nn.BatchNorm2D(hidden_channels)
        self.block = CSModule(hidden_channels, hidden_channels, kernel_sizes, dilations,expansion=1.0)
        self.norm2 = nn.BatchNorm2D(hidden_channels)
        self.mlp = Mlp(in_features=hidden_channels,
                       hidden_features=int(hidden_channels * ffn_scale),
                       drop=drop_path_rate)
                
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        layer_scale_init_value = paddle.full(
            [hidden_channels, 1, 1], fill_value=1e-2, dtype="float32")
        self.layer_scale_1 = paddle.create_parameter(
            [hidden_channels, 1, 1], "float32", attr=Assign(layer_scale_init_value))
        self.layer_scale_2 = paddle.create_parameter(
            [hidden_channels, 1, 1], "float32", attr=Assign(layer_scale_init_value))

    def forward(self,x):

        x = x + self.drop_path(self.layer_scale_1 *self.block( self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 *self.mlp( self.norm2(x)))
        return x
    

class DownSampleConv(nn.Layer):
    def __init__(self, in_planes,out_planes):
        super(DownSampleConv, self).__init__()
       
        self.branch_1 = nn.Sequential(
            BasicConv(in_planes,out_planes,3,stride=2,padding=1,relu=True,bn=True)
        )
    def forward(self, x):
        return self.branch_1(x)
    
class Stem(nn.Layer):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super(Stem, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(out_channels // 2),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2D(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(out_channels//2),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2D(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv1_3(self.conv1_2(self.conv1_1(x)))
  

class UpsamplingBottleneck(nn.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super(UpsamplingBottleneck, self).__init__()
        self.conv1 = BasicConv(in_channels,out_channels,kernel_size=1,relu=True,bn=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = BasicConv(out_channels,out_channels,kernel_size=1,relu=True,bn=True)
    def forward(self, x):
        x = self.conv1(x)
        max_up = self.up(x)
        return self.conv2(max_up)
 
 

class CrackStage(nn.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_num:int,
            stage_num:int=2,
            drop_path:float=0.1
    ):
        super(CrackStage, self).__init__()
        # self.in_c = in_channels
        # self.out_c = out_channels
        self.stage_num = stage_num
        self.down_sample = DownSampleConv(in_channels,out_channels) 
        # self.blocks = nn.LayerList( [LRFDBlock(out_channels,out_channels) for i in range(block_num)])
        if(block_num == 1):
            self.blocks = nn.Sequential(
                CSBlock(out_channels,drop_path_rate=drop_path)
            )
        elif(block_num == 2):
            self.blocks = nn.Sequential(
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path)
            )
        elif(block_num == 3):
            self.blocks = nn.Sequential(
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path)
            )
        elif(block_num == 4):
            self.blocks = nn.Sequential(
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path)
            )
        elif(block_num == 5):
            self.blocks = nn.Sequential(
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path)
            )
        else:
            self.blocks = nn.Sequential(
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path),
                CSBlock(out_channels,drop_path_rate=drop_path)
            )
    def forward(self, x):
        if(self.stage_num != 1):
            x = self.down_sample(x)
        return self.blocks(x)

class CrackScopeNet_Seg(nn.Layer):
    def __init__(
            self,
            base: int,
            block_num:Sequence[int]=(2,2,4),
            feat_channels:Sequence[int]=(32,64,128),
            num_classes=2, # number of classes
            pretrained=None, # pretrained model
            dropout_rate:float=0.1
    ):
        super(CrackScopeNet_Seg, self).__init__()
        self.pretrained = pretrained
        self.feat_channels = feat_channels
        self.stem = Stem(3,base)
        self.stage1 = CrackStage(feat_channels[0],feat_channels[0],block_num[0],stage_num=1,drop_path=dropout_rate)   #64 
        self.stage2 = CrackStage(feat_channels[0],feat_channels[1],block_num[1],drop_path=dropout_rate)  #128
        self.stage3 = CrackStage(feat_channels[1],feat_channels[2],block_num[2],drop_path=dropout_rate)  #256   1/8

        self.up2 = UpsamplingBottleneck(feat_channels[2],feat_channels[1])   # 128---64

        
        self.cat1 = CSModule_Upsampling(feat_channels[1]*2,feat_channels[1]*2)
        self.dw3_1 = DWConv(feat_channels[1]*2,feat_channels[1]*2)
        self.up3 = UpsamplingBottleneck(feat_channels[1]*2,feat_channels[0]) 

        self.cat = CSModule_Upsampling(feat_channels[0]*2,feat_channels[0]*2)   # feat1*3  
        self.dw3 = DWConv(feat_channels[0]*2,feat_channels[0]*2)
        # 跟一个dw卷积
        self.seghead = AuxSegHead(feat_channels[0]*2,feat_channels[0],num_classes)
        
        self.aux_head1 = AuxSegHead(feat_channels[2],feat_channels[2]//2,num_classes)   # 1/16
        self.aux_head2 = AuxSegHead(feat_channels[1],feat_channels[1]//2,num_classes)   # 1/8
        self.aux_head3 = AuxSegHead(feat_channels[0],feat_channels[0]//2,num_classes)   # 1/4
        self.aux_head4 = AuxSegHead(feat_channels[0]*2,feat_channels[0],num_classes)   # 1/4
        
        self.init_weight()
        
    def forward(self, x):
        h, w = paddle.shape(x)[2:]
        logit_list = []
        x = self.stem(x)

        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        if self.training:
            aux1 = self.aux_head1(feat3)
            aux2 = self.aux_head2(feat2)
            aux3 = self.aux_head3(feat1)
        
            up2 = self.up2(feat3)
            up3 = self.up3(self.dw3_1(self.cat1(paddle.concat([up2,feat2],axis=1))))

   
            cat = self.dw3(self.cat(paddle.concat([up3,feat1],axis=1)))

            last_out = self.seghead(cat)
            logit_list = [last_out,aux1,aux2,aux3]
            logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
            return  logit_list
        else:
            up2 = self.up2(feat3)
            up3 = self.up3(self.dw3_1(self.cat1(paddle.concat([up2,feat2],axis=1))))
            cat = self.dw3(self.cat(paddle.concat([up3,feat1],axis=1)))

            last_out = self.seghead(cat)
            logit_list = [last_out]
            logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
            return  logit_list
        
    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                    if isinstance(m, nn.Conv2D):
                        param_init.kaiming_normal_init(m.weight)
                    elif isinstance(m, nn.BatchNorm2D):
                        param_init.constant_init(m.weight, value=1)
                        param_init.constant_init(m.bias, value=0)



        
# seg head
class SegHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes):
        super(SegHead, self).__init__()
        self.trans_conv = nn.Sequential(
                nn.Conv2DTranspose(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2D(interplanes),
                # nn.Silu()
                nn.PReLU()
                )
        self.conv1 = BasicConv(interplanes,interplanes,kernel_size=3,stride=1, padding=1)
        self.conv2 = BasicConv(interplanes,outplanes,kernel_size=1)

    def forward(self, x):
        x = self.trans_conv(x)
        out = self.conv2(self.conv1(x))
        return out
       

     
     
class AuxSegHead(nn.Layer):
    def __init__(self, in_dim, mid_dim, num_classes=2):
        super().__init__()

        self.conv_3x3 = nn.Sequential(
            layers.ConvBNReLU(in_dim, mid_dim, 3), nn.Dropout(0.1))

        self.conv_1x1 = nn.Conv2D(mid_dim, num_classes, 1, 1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        return conv2
      
 
@manager.MODELS.add_component
def CrackScopeNet(num_classes=2):
    model = CrackScopeNet_Seg(32,(3,3,4),(32,64,128),dropout_rate=0.1)  # 

@manager.MODELS.add_component
def CrackScopeNet_Large(num_classes=2):
    model = CrackScopeNet_Seg(64,(3,3,3),(64,128,160),dropout_rate=0.1)
    return model

# if __name__ == "__main__":
#     model = CrackNet_Seg(32,(2,2,4),(32,64,128))
#     x = paddle.randn([1, 3, 400, 400])
#     out = model(x)
#     print(out[0].shape)
#     print(model.dw3)

#     paddle.flops(model, input_size=(1, 3, 400, 400))
        