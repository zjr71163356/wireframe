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

    def postprocess(self) -> Any:
        def process(im: torch.Tensor) -> torch.Tensor:
            im = im * self.std + self.mean
            im = im.permute(1, 2, 0).cpu()  # Keep as tensor
            return im

        return process

    def postprocessLine(self) -> Any:
        def process(im: torch.Tensor) -> torch.Tensor:
            if isinstance(im, np.ndarray):  # Compatibility check
                im = torch.from_numpy(im)
            im = im.permute(1, 2, 0).cpu()  # Keep as tensor
            return im

        return process


def getInstance(info: Dict[str, Any], opt: Any, split: str) -> IndoorDist:
    return IndoorDist(info, opt, split)
