import math
import random
import torch
import torch.nn.functional as F
from torch import Tensor


# ipt is a tensor with shape (channel, height, width)
# xml is a tensor with shape (height, width)

def addNoise(ipt: Tensor, miu: float, std: float) -> Tensor:
    noise = torch.normal(miu, std, ipt.shape)
    return ipt + noise


def scaleRGB(ipt: Tensor) -> Tensor:
    return ipt / 255.0


def unScaleRGB(ipt: Tensor) -> Tensor:
    opt = ipt * 255
    opt = opt.to(torch.uint8)
    return opt

def normalize(ipt: Tensor, mean: list[float], std: list[float]) -> Tensor:
    # 确保 mean 和 std 是与 ipt 相同设备上的张量
    mean = torch.tensor(mean, device=ipt.device).view(-1, 1, 1)
    std = torch.tensor(std, device=ipt.device).view(-1, 1, 1)

    # 归一化每个通道
    ipt = (ipt - mean) / std
    return ipt

def unNormalize(ipt: Tensor, mean: list[float], std: list[float]) -> Tensor:
    # 确保 mean 和 std 是与 ipt 相同设备上的张量
    mean = torch.tensor(mean, device=ipt.device).view(-1, 1, 1)
    std = torch.tensor(std, device=ipt.device).view(-1, 1, 1)

    # 反归一化每个通道
    ipt = ipt * std + mean
    return ipt



def randomFlip(ipt: Tensor, xml: Tensor) -> tuple[Tensor, Tensor]:
    if random.uniform(0, 1) > 0.5:
        ipt = torch.flip(ipt, dims=[2])  # Flip along the width dimension
        xml = torch.flip(xml, dims=[1])  # Flip along the width dimension
    return ipt, xml


def randomCrop(ipt: Tensor, xml: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
    origH = ipt.shape[1]
    origW = ipt.shape[2]
    newH = size[0]
    newW = size[1]

    # Randomly select the starting point
    startH = random.randint(0, origH - newH)
    startW = random.randint(0, origW - newW)

    # Crop the image and XML
    ipt = ipt[:, startH: startH + newH, startW: startW + newW]
    xml = xml[startH: startH + newH, startW: startW + newW]

    return ipt, xml


def randomSizeCrop(ipt: Tensor, xml: Tensor, LowBound: float) -> tuple[Tensor, Tensor]:
    newH = math.floor(random.uniform(LowBound, 1) * ipt.shape[1])
    newH = newH - (newH % 8)  # Ensure divisible by 8
    newW = math.floor(random.uniform(LowBound, 1) * ipt.shape[2])
    newW = newW - (newW % 8)  # Ensure divisible by 8
    return randomCrop(ipt, xml, (newH, newW))
