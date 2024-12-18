import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np 
import torch
from torchvision import transforms
from utils import transform
# 데이터셋 준비

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

# 데이터셋 로드
dataset = PennFudanDataset('PennFudanPed', transforms=transform)
dataset_test = PennFudanDataset('PennFudanPed', transforms=transform)

# 데이터셋 분할
indices = torch.randperm(len(dataset)).tolist()
split_index = int(len(indices) * 0.8)
dataset = torch.utils.data.Subset(dataset, indices[:split_index])
dataset_test = torch.utils.data.Subset(dataset_test, indices[split_index:])

# 데이터 로더 생성
def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
