import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from bn import BN
from conv import Conv
from relu import ReLU

# commen params
# conv
padding = 2
stride = 2

# bn
gamma = 1.0
beta = 0.0
eps = 1e-5
momentum = 0.1

# input
np.random.seed(1)
A = np.random.randn(20,3,128,128)
W = np.random.randn(16,3,3,3)


# ===============
start = time.time()
# conv
conv_layer = Conv(stride, padding)
conv_Z = conv_layer.forward(A, W)

# bn
bn_layer = BN(gamma, beta, eps)
bn_Z = bn_layer.forward(conv_Z)

# relu
relu_layer = ReLU()
relu_Z = relu_layer.forward(bn_Z)

# conv
_, C, _, _ = relu_Z.shape
W2 = np.random.randn(32,C,3,3)
conv_layer2 = Conv(stride, padding)
conv_Z_2 = conv_layer2.forward(relu_Z, W2)

# bn
bn_layer_2 = BN(gamma, beta, eps)
bn_Z_2 = bn_layer_2.forward(conv_Z_2)

# relu
relu_layer_2 = ReLU()
relu_Z_2 = relu_layer_2.forward(bn_Z_2)

end = time.time()

print('customized NN running time: ', end - start)
#=============

A = torch.from_numpy(A)
W = torch.from_numpy(W)
W2 = torch.from_numpy(W2)

start = time.time()
# conv
conv_Z_gt = F.conv2d(A, W, bias=None, stride=stride, padding=padding, dilation=1, groups=1)
print('conv diff: ', torch.sum(torch.abs(conv_Z_gt - conv_Z)))

# bn
(_, channel, _, _) = conv_Z_gt.shape
bn = nn.BatchNorm2d(channel, eps=eps, momentum=momentum, affine=False, track_running_stats=False)
bn_Z_gt = bn(conv_Z_gt.float())
print('bn diff: ', torch.sum(torch.abs(bn_Z_gt - bn_Z)))

# relu
relu=nn.ReLU(inplace=True)
relu_Z_gt =relu(bn_Z_gt)
print('relu diff: ', torch.sum(torch.abs(relu_Z_gt - relu_Z)))

# conv
conv_Z_gt_2 = F.conv2d(relu_Z_gt.double(), W2, bias=None, stride=stride, padding=padding, dilation=1, groups=1)
print('conv2 diff: ', torch.sum(torch.abs(conv_Z_gt_2 - conv_Z_2)))

# bn
(_, channel_2, _, _) = conv_Z_gt_2.shape
bn_2 = nn.BatchNorm2d(channel_2, eps=eps, momentum=momentum, affine=False, track_running_stats=False)
bn_Z_gt_2 = bn_2(conv_Z_gt_2.float())
print('bn2 diff: ', torch.sum(torch.abs(bn_Z_gt_2 - bn_Z_2)))

# relu
relu2=nn.ReLU(inplace=True)
relu_Z_gt_2 =relu2(bn_Z_gt_2)
print('relu2 diff: ', torch.sum(torch.abs(relu_Z_gt_2 - relu_Z_2)))

end = time.time()

print('pytorch NN running time: ', end - start)