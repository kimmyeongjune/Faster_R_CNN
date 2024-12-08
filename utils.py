import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image

# 데이터 resize 및 tensor로 변환

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size  # (width, height)

    def __call__(self, image, target):
        # 원본 이미지 크기
        w, h = image.size

        # 새로운 이미지 크기
        new_w, new_h = self.size

        # 이미지 리사이즈
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # 바운딩 박스 좌표 조정
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes * torch.tensor([new_w / w, new_h / h, new_w / w, new_h / h])
            target["boxes"] = boxes

        # 마스크 리사이즈
        if "masks" in target:
            masks = target["masks"]
            masks = masks.unsqueeze(0)  # (N, H, W) -> (1, N, H, W)
            masks = nn.functional.interpolate(masks, size=(new_h, new_w), mode='nearest')
            masks = masks.squeeze(0)
            target["masks"] = masks

        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = transforms.functional.to_tensor(image)
        return image, target

transform = Compose([
    Resize((800, 800)),
    ToTensor()
])