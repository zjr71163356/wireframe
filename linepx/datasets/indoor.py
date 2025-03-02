import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Any, Dict, Tuple


class IndoorDist(Dataset):
    def __init__(self, imageInfo: Dict[str, Any], opt: Any, split: str) -> None:
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.preprocessLine = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Precompute the mean and std tensor for use later
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=torch.device('cpu')).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=torch.device('cpu')).view(3, 1, 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        imgPath = str(self.imageInfo['imagePath'][index])
        image = Image.open(imgPath).convert('RGB')
        image = self.preprocess(image)

        linePath = str(self.imageInfo['linePath'][index])
        line = Image.open(linePath).convert('L')
        line = self.preprocessLine(line)

        # Use torch's `interpolate` more efficiently with pre-calculated device tensors
        line = torch.nn.functional.interpolate(line.unsqueeze(0), size=(self.opt.imgDim, self.opt.imgDim),
                                               mode='bilinear', align_corners=False).squeeze(0)

        imgName = os.path.basename(imgPath).replace('_rgb.png', '')
        return image, line, imgName

    def __len__(self) -> int:
        return len(self.imageInfo['imagePath'])

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将图像张量转换为适合 OpenCV 的 NumPy 数组。

        参数:
        - tensor: PyTorch 张量，形状为 [C, H, W]。

        返回:
        - img: NumPy 数组，类型为 uint8，形状为 [H, W, C]。
        """
        tensor = tensor.detach().cpu()
        img = tensor.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = (img * self.std.numpy().transpose(1, 2, 0) + self.mean.numpy().transpose(1, 2, 0)) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def postprocessLine(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将线条张量转换为适合 OpenCV 的 NumPy 数组。

        参数:
        - tensor: PyTorch 张量，形状可能为 [1, H, W] 或 [H, W]。

        返回:
        - line_img: NumPy 数组，类型为 uint8，形状为 [H, W]。
        """
        tensor = tensor.detach().cpu()
        line_img = tensor.numpy()
        if line_img.ndim == 3 and line_img.shape[0] == 1:
            line_img = line_img.squeeze(0)  # [1, H, W] -> [H, W]
        line_img = (line_img * 255).astype(np.uint8)
        return line_img

def getInstance(info: Dict[str, Any], opt: Any, split: str) -> IndoorDist:
    return IndoorDist(info, opt, split)
