import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn

class SuperResolutionNet(nn.Module):
    # 图片的放大比例是写死在模型里的
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
# 现有实现插值的 PyTorch 算子有一套规定好的映射到 ONNX Resize 算子的方法，
# 这些映射出的 Resize 算子的 scales 只能是常量，无法满足我们的需求。
# 我们得自己定义一个实现插值的 PyTorch 算子，然后让它映射到一个我们期望的 ONNX Resize 算子上。

def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
    
    state_dict = torch.load('srcnn.pth')['state_dict']
    
    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()
input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch.png", torch_output)