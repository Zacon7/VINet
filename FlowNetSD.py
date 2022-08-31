# FlowNetSD used in the VINNet


import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from submodules import *
'Parameter count = 45,371,666'

class FlowNetSD(nn.Module):
    def __init__(self, batchNorm=True):
        super(FlowNetSD,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  6,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)
        self.conv3   = conv(self.batchNorm, 128,  256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.inter_conv5 = i_conv(self.batchNorm,  1026,   512)
        self.inter_conv4 = i_conv(self.batchNorm,  770,   256)
        self.inter_conv3 = i_conv(self.batchNorm,  386,   128)
        self.inter_conv2 = i_conv(self.batchNorm,  194,   64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')



    def forward(self, x):
        print("Flownet input shape:", x.shape)
        out_conv0 = self.conv0(x)
        print("out_conv0:", out_conv0.shape)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        print("out_conv1:", out_conv1.shape)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        print("out_conv2:", out_conv2.shape)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        print("out_conv3:", out_conv3.shape)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        print("out_conv4:", out_conv4.shape)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        print("out_conv5:", out_conv5.shape)
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        print("out_conv6:", out_conv6.shape)

        flow6       = self.predict_flow6(out_conv6)
        print("flow6:", flow6.shape)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        print("flow6_up:", flow6_up.shape)
        out_deconv5 = self.deconv5(out_conv6)
        print("out_deconv5:", out_deconv5.shape)
       
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        print("concat5:", concat5.shape)

        return concat5

        ## --- TÄSTÄ POIKKI ETTÄ SAADAAN SHAPE BATCH x 1024 x 6 x 8 ----
'''
        out_interconv5 = self.inter_conv5(concat5)
        print("out_interconv5:", out_interconv5.shape)
        flow5       = self.predict_flow5(out_interconv5)
        print("flow5:", flow5.shape)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        print("flow5_up:", flow5_up.shape)
        out_deconv4 = self.deconv4(concat5)
        print("out_deconv4:", out_deconv4.shape)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        print("concat4:", concat4.shape)
        out_interconv4 = self.inter_conv4(concat4)
        print("out_interconv4:", out_interconv4.shape)
        flow4       = self.predict_flow4(out_interconv4)
        print("flow4:", flow4.shape)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        print("flow4_up:", flow4_up.shape)
        out_deconv3 = self.deconv3(concat4)
        print("out_deconv3:", out_deconv3.shape)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        print("concat3:", concat3.shape)
        out_interconv3 = self.inter_conv3(concat3)
        print("out_interconv3:", out_interconv3.shape)
        flow3       = self.predict_flow3(out_interconv3)
        print("flow3:", flow3.shape)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        print("flow3_up:", flow3_up.shape)
        out_deconv2 = self.deconv2(concat3)
        print("out_deconv2:", out_deconv2.shape)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        print("concat2:", concat2.shape)
        out_interconv2 = self.inter_conv2(concat2)
        print("out_interconv2:", out_interconv2.shape)
        flow2 = self.predict_flow2(out_interconv2)
        print("flow2:", flow2.shape)

        print("\nFLOWNET WORKS!!\n")

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2,
'''