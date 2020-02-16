# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
import numpy as np

class BatchNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, decay=0.9):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=num_features, 
                                         momentum=1-decay, 
                                         eps=epsilon, 
                                         affine=True, 
                                         track_running_stats=True)

    def forward(self, x):
        return self.batch_norm(x)

class Conv2d(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=5, stride=2, stddev=0.02):
        super(Conv2d, self).__init__()
        #for same padding
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.same_padding = nn.ConstantPad2d((ka,kb,ka,kb),0)
        self.conv2d = nn.Conv2d(input_ch, output_ch, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                bias=True)
        nn.init.normal_(self.conv2d.weight, std=stddev)
        torch.clamp(self.conv2d.weight, -2*stddev, 2*stddev)
        nn.init.constant_(self.conv2d.bias, 0.0)

    def forward(self, x):
        x = self.same_padding(x)
        return self.conv2d(x)

class Deconv2d(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=5, stride=2, stddev=0.02):
        super(Deconv2d, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(input_ch, output_ch, 
                                           kernel_size=kernel_size, 
                                           stride=stride, 
                                           padding=2, 
                                           output_padding=1, 
                                           bias=True)
        nn.init.normal_(self.deconv2d.weight, std=stddev)
        nn.init.constant_(self.deconv2d.bias, 0.0)
    
    def forward(self, x):
        return self.deconv2d(x)

class Lrelu(nn.Module):
    def __init__(self, leak=0.2):
        super(Lrelu, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=leak)

    def forward(self, x):
        return self.lrelu(x)

class FC(nn.Module):
    def __init__(self, num_features, output_size, stddev=0.02):
        super(FC, self).__init__()
        self.num_features = num_features
        self.fc = nn.Linear(num_features, output_size)
        nn.init.normal_(self.fc.weight, std=stddev)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, x):
        x = x.view(-1, self.num_features)
        return self.fc(x)

class Embedding(nn.Module):
    def __init__(self, size, dimension, stddev=0.01):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(size, dimension)
        nn.init.normal_(self.embedding.weight, std=stddev)
    
    def forward(self, ids):
        return self.embedding(ids.long())

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, labels_num, output_filters):
        super(ConditionalInstanceNorm, self).__init__()
        self.scale = nn.Embedding(labels_num, output_filters)
        nn.init.constant_(self.scale.weight, 1.0)
        self.shift = nn.Embedding(labels_num, output_filters)
        nn.init.constant_(self.shift.weight, 0.0)

    def forward(self, x, ids):
        shape = x.shape
        batch_size, output_filters = shape[0], shape[1]

        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = x.std(dim=(2, 3), keepdim=True)

        norm = (x - mu) / torch.sqrt(sigma + 1e-5)

        batch_scale = self.scale(ids.long())
        batch_scale = batch_scale.view(batch_size, output_filters, 1, 1)
        batch_shift = self.shift(ids.long())
        batch_shift = batch_shift.view(batch_size, output_filters, 1, 1)

        z = norm * batch_scale + batch_shift
        return z

class InstanceNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, decay=0.9):
        super(InstanceNorm, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features=num_features, 
                                               momentum=1-decay, 
                                               eps=epsilon, 
                                               affine=False, 
                                               track_running_stats=False)

    def forward(self, x):
        return self.instance_norm(x)

class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, labels_num, eps=1e-4, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
        self.gamma_embed = SpectralNorm(nn.Linear(labels_num, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(labels_num, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v, eps=1e-4):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = self.l2normalize(torch.matmul(_w.t(), u))
            u = self.l2normalize(torch.matmul(_w, v))

        sigma = u.dot((_w).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

