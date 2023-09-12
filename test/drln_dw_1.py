import torch
import torch.nn as nn

import opsdw as ops
import torch.nn.functional as F

def make_model(args, parent=False):
    return DRLN(args)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel , channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel , channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction)*3, channel , 3, 1, 1)
        # self.c5 = ops.BasicBlockSig((channel // reduction)*2, channel , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        # c_out1 = torch.cat([c1, c2], dim=1) computing size gap is small
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1, split = 0.25):
        super(Block, self).__init__()
        self.distilled_channels = int(in_channels * split)
        self.remaining_channels = in_channels - self.distilled_channels
        self.r0 = ops.BasicBlock_dw_1(in_channels, in_channels)
        self.r1 = ops.BasicBlock_dw_1(self.remaining_channels, in_channels)
        self.r2 = ops.BasicBlock_dw_1(self.remaining_channels, in_channels)
        self.r3 = ops.BasicBlock_dw(self.remaining_channels, self.distilled_channels)
        self.g1 = ops.BasicBlock_dw(in_channels, out_channels, 1, 1, 0)
        self.g = ops.BasicBlock_dw(in_channels*8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 =  x

        out0 = self.r0(c0)
        d0, r0 = torch.split(out0, (self.distilled_channels, self.remaining_channels), dim=1)
        out1 = self.r1(r0)
        d1, r1 = torch.split(out1, (self.distilled_channels, self.remaining_channels), dim=1)
        out2 = self.r2(r1)
        d2, r2 = torch.split(out2, (self.distilled_channels, self.remaining_channels), dim=1)
        d3 = self.r3(r2)
                
        c0 = torch.cat([d0, d1, d2, d3], dim=1)
        # c0 = r2 + c0
        g1 = self.g1(c0)
        out = self.ca(g1)
        return out
        

class DRLN(nn.Module):
    def __init__(self, scale = 3):
        super(DRLN, self).__init__()

        self.scale = scale
        chs = 36

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.head = nn.Conv2d(3, chs, 3, 1, 1)

        self.b1 = Block(chs, chs)
        self.b2 = Block(chs, chs)
        self.b3 = Block(chs, chs)
        self.b4 = Block(chs, chs)
        self.b5 = Block(chs, chs)
        self.b6 = Block(chs, chs)
        self.b7 = Block(chs, chs)
        self.b8 = Block(chs, chs)


        self.c1 = ops.BasicBlock_dw(chs, chs, 3, 1, 1)
        self.c2 = ops.BasicBlock_dw(chs, chs, 3, 1, 1)
        self.c3 = ops.BasicBlock_dw(chs*2, chs, 3, 1, 1)
        self.c4 = ops.BasicBlock_dw(chs*3, chs, 3, 1, 1)
        self.c5 = ops.BasicBlock_dw(chs*2, chs, 3, 1, 1)
        self.c6 = ops.BasicBlock_dw(chs*3, chs, 3, 1, 1)
        self.c7 = ops.BasicBlock_dw(chs*2, chs, 3, 1, 1)
        self.c8 = ops.BasicBlock_dw(chs*3, chs, 3, 1, 1)


        self.upsample = ops.UpsampleBlock(chs, self.scale , multi_scale=False)
        self.cca = CALayer(chs * 12)
        self.convert1 = nn.Conv2d(chs*12, chs, 1, 1, 0)
        self.convert = ops.ConvertBlock(chs, chs, 20)
        self.tail = nn.Conv2d(3, 3, 1, 1, 0)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        
        b2 = self.b2(b1)
        o2 = self.c2(b2)
        a0 = o2 + o0
        
        b3 = self.b3(b2)
        c3 = torch.cat([o2, b3], dim=1)
        o3 = self.c3(c3)
        

        b4 = self.b4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.c4(c4)
        a1 = o4 + a0
 
        b5 = self.b5(a1)
        c5 = torch.cat([o4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1

        b7 = self.b7(a2)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)
        a3 = o8 + a2

        b_out =  a3 + x
        out = self.upsample(b_out, scale = self.scale )

        out = self.tail(out)
        f_out = self.add_mean(out)

        return f_out
    
