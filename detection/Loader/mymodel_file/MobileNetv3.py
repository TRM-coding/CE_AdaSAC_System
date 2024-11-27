import torch
import torch.nn as nn

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias,isAct=None):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.isAct = isAct
        
        if isAct == "hardswish":
            self.act = nn.Hardswish()
        elif isAct == "relu":
            self.act = nn.ReLU(inplace=True)
            
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.isAct == "hardswish" or self.isAct == "relu":
            x = self.act(x)
        return x
    

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.scale_activation(x)
        return x
    


class MobileNetv3(nn.Module):
    def __init__(self):
        super(MobileNetv3, self).__init__()
        
        self.conv0 = Conv2dNormActivation(3,16,3,2,1,1,False,"hardswish")
        
        self.conv1_0 = Conv2dNormActivation(16,16,3,1,1,16,False,"relu")
        self.conv1_1 = Conv2dNormActivation(16,16,1,1,0,1,False)
        
        self.conv2_0 = Conv2dNormActivation(16,64,1,1,1,1,False,"relu")
        self.conv2_1 = Conv2dNormActivation(64,64,3,2,1,64,False,"relu")
        self.conv2_2 = Conv2dNormActivation(64,24,1,1,0,1,False)
        
        self.conv2_0 = Conv2dNormActivation(16,64,1,1,1,1,False,"relu")
        self.conv2_1 = Conv2dNormActivation(64,64,3,2,1,64,False,"relu")
        self.conv2_2 = Conv2dNormActivation(64,24,1,1,0,1,False)
        
        self.conv3_0 = Conv2dNormActivation(24,72,1,1,0,1,False,"relu")
        self.conv3_1 = Conv2dNormActivation(72,72,3,1,1,72,False,"relu")
        self.conv3_2 = Conv2dNormActivation(72,24,1,1,0,1,False)
        
        self.conv4_0 = Conv2dNormActivation(24,72,1,1,0,1,False,"relu")
        self.conv4_1 = Conv2dNormActivation(72,72,5,2,2,72,False,"relu")
        self.squeeze4_2 = SqueezeExcitation(72,24)
        self.conv4_3 = Conv2dNormActivation(72,40,1,1,0,1,False)
        
        self.conv5_0 = Conv2dNormActivation(40,120,1,1,0,1,False,"relu")
        self.conv5_1 = Conv2dNormActivation(120,120,5,1,2,120,False,"relu")
        self.squeeze5_2 = SqueezeExcitation(120,32)
        self.conv5_3 = Conv2dNormActivation(120,40,1,1,0,1,False)
        
        self.conv6_0 = Conv2dNormActivation(40,120,1,1,0,1,False,"relu")
        self.conv6_1 = Conv2dNormActivation(120,120,3,2,1,120,False,"relu")
        self.squeeze6_2 = SqueezeExcitation(120,32)
        self.conv6_3 = Conv2dNormActivation(120,40,1,1,0,1,False)
        
        self.conv7_0 = Conv2dNormActivation(40,240,1,1,0,1,False,"hardswish")
        self.conv7_1 = Conv2dNormActivation(240,240,3,2,1,240,False,"hardswish")
        self.conv7_2 = Conv2dNormActivation(240,80,1,1,0,1,False)
        
        self.conv8_0 = Conv2dNormActivation(80,200,1,1,0,1,False,"hardswish")
        self.conv8_1 = Conv2dNormActivation(200,200,3,1,1,200,False,"hardswish")
        self.conv8_2 = Conv2dNormActivation(200,80,1,1,0,1,False)
        
        self.conv9_0 = Conv2dNormActivation(80,184,1,1,0,1,False,"hardswish")
        self.conv9_1 = Conv2dNormActivation(184,184,3,1,1,184,False,"hardswish")
        self.conv9_2 = Conv2dNormActivation(184,80,1,1,0,1,False)
        
        self.conv10_0 = Conv2dNormActivation(80,184,1,1,0,1,False,"hardswish")
        self.conv10_1 = Conv2dNormActivation(184,184,3,1,1,184,False,"hardswish")
        self.conv10_2 = Conv2dNormActivation(184,80,1,1,0,1,False)
        
        self.conv11_0 = Conv2dNormActivation(80,480,1,1,0,1,False,"hardswish")
        self.conv11_1 = Conv2dNormActivation(480,480,3,1,1,480,False,"hardswish")
        self.squeeze11_2 = SqueezeExcitation(480,120)
        self.conv11_3 = Conv2dNormActivation(480,112,1,1,0,1,False)
        
        self.conv12_0 = Conv2dNormActivation(112,672,1,1,0,1,False,"hardswish")
        self.conv12_1 = Conv2dNormActivation(672,672,3,1,1,672,False,"hardswish")
        self.squeeze12_2 = SqueezeExcitation(672,168)
        self.conv12_3 = Conv2dNormActivation(672,112,1,1,0,1,False)
        
        self.conv13_0 = Conv2dNormActivation(112,672,1,1,0,1,False,"hardswish")
        self.conv13_1 = Conv2dNormActivation(672,672,5,2,2,672,False,"hardswish")
        self.squeeze13_2 = SqueezeExcitation(672,168)
        self.conv13_3 = Conv2dNormActivation(672,160,1,1,0,1,False)
        
        self.conv14_0 = Conv2dNormActivation(160,960,1,1,0,1,False,"hardswish")
        self.conv14_1 = Conv2dNormActivation(960,960,5,1,2,960,False,"hardswish")
        self.squeeze14_2 = SqueezeExcitation(960,240)
        self.conv14_3 = Conv2dNormActivation(960,160,1,1,0,1,False)
        
        
        self.conv15_0 = Conv2dNormActivation(160,960,1,1,0,1,False,"hardswish")
        self.conv15_1 = Conv2dNormActivation(960,960,5,1,2,960,False,"hardswish")
        self.squeeze15_2 = SqueezeExcitation(960,240)
        self.conv15_3 = Conv2dNormActivation(960,160,1,1,0,1,False)
        
        self.conv16_0 = Conv2dNormActivation(160,960,1,1,0,1,False,"hardswish")
        
        self.avgpool17 = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.fc18_1 = nn.Linear(960, 1280)
        self.hs18 = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc18_2 = nn.Linear(1280, 1000)
        
    def forward(self, x):
        x = self.conv0(x)  
        x = self.conv1_0(x)  
        x = self.conv1_1(x)
        
        x = self.conv2_0(x) 
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        
        x = self.conv3_0(x)  
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        
        x = self.conv4_0(x)
        x = self.conv4_1(x)  
        x = self.squeeze4_2(x)
        x = self.conv4_3(x)
        
        x = self.conv5_0(x)  
        x = self.conv5_1(x)
        x = self.squeeze5_2(x)
        x = self.conv5_3(x)
        
        x = self.conv6_0(x)  
        x = self.conv6_1(x)
        x = self.squeeze6_2(x)
        x = self.conv6_3(x)
        
        x = self.conv7_0(x)  
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        
        x = self.conv8_0(x)  
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        
        x = self.conv9_0(x)  
        x = self.conv9_1(x)
        x = self.conv9_2(x)
        
        x = self.conv10_0(x)  
        x = self.conv10_1(x)
        x = self.conv10_2(x)
        
        x = self.conv11_0(x)  
        x = self.conv11_1(x)
        x = self.squeeze11_2(x)
        x = self.conv11_3(x)
        
        x = self.conv12_0(x)  
        x = self.conv12_1(x)
        x = self.squeeze12_2(x)
        x = self.conv12_3(x)
        
        x = self.conv13_0(x)  
        x = self.conv13_1(x)
        x = self.squeeze13_2(x)
        x = self.conv13_3(x)
        
        x = self.conv14_0(x)  
        x = self.conv14_1(x)
        x = self.squeeze14_2(x)
        x = self.conv14_3(x)
        
        x = self.conv15_0(x)  
        x = self.conv15_1(x)
        x = self.squeeze15_2(x)
        x = self.conv15_3(x)
        
        x = self.conv16_0(x)  
        x = self.avgpool17(x)
        x = self.flatten(x)
        
        x = self.fc18_1(x)
        x = self.hs18(x)
        x = self.dropout(x)
        x = self.fc18_2(x)
        
        return x
