# ==============================================================================
# Tên file: model.py
# Công dụng: Định nghĩa kiến trúc mạng nơ-ron và hàm mất mát ArcFace.
#            - backbone IResNet (iresnet34, iresnet50, iresnet100) cải tiến từ ResNet
#              (dùng PReLU, Pre-activation BN, loại bỏ MaxPool).
#            - header ArcMarginProduct áp dụng biên phạt góc góc m (Additive Angular Margin)
#              trực tiếp vào target logit để nén chặt vector 512 chiều.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.update_margin(m)

    def update_margin(self, m):
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Cosine similarity (cast to float32 for numerical stability in AMP)
        cosine = F.linear(F.normalize(input.float()), F.normalize(self.weight.float()))
        # Numerical stability: clamp to avoid NaN
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # Prevent sqrt(0) which causes NaN/Inf gradients in AMP
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7))
        # Phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # Target logit penalty
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # Cast back to input dtype for AMP compatibility
        return output.type_as(input)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(IResNetBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IResNetBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IResNetBlock")
        
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class IResNet(nn.Module):
    def __init__(self, block, layers, dropout=0, num_features=512, groups=1, width_per_group=64, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution (no maxpooling in IResNet)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                if m.weight is not None and m.weight.requires_grad:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)

        return x

def iresnet34(num_features=512, **kwargs):
    return IResNet(IResNetBlock, [3, 4, 6, 3], num_features=num_features, **kwargs)

def iresnet50(num_features=512, **kwargs):
    return IResNet(IResNetBlock, [3, 4, 14, 3], num_features=num_features, **kwargs)

def iresnet100(num_features=512, **kwargs):
    return IResNet(IResNetBlock, [3, 13, 30, 3], num_features=num_features, **kwargs)

def get_backbone(name='iresnet50', embedding_size=512, dropout=0):
    if name == 'iresnet34':
        return iresnet34(num_features=embedding_size, dropout=dropout)
    elif name == 'iresnet50':
        return iresnet50(num_features=embedding_size, dropout=dropout)
    elif name == 'iresnet100':
        return iresnet100(num_features=embedding_size, dropout=dropout)
    else:
        raise ValueError(f"Unknown backbone name: {name}")