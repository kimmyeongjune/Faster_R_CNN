import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from PIL import Image
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import urllib.request
import tarfile

from utils import Compose, Resize, ToTensor
from dataset import PennFudanDataset, dataset, dataset_test, data_loader, data_loader_test
from model import backbone, anchor_generator, roi_pooler, FasterRCNNHead, RPNHead, FasterRCNN, rpn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 다운로드


# 데이터셋 다운로드 경로
dataset_url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
dataset_dir = 'PennFudanPed'

# 데이터셋이 없을 경우 다운로드 및 압축 해제
if not os.path.exists(dataset_dir):
    print("데이터셋을 다운로드 중입니다...")
    urllib.request.urlretrieve(dataset_url, 'PennFudanPed.zip')
    print("압축 해제 중입니다...")
    import zipfile
    with zipfile.ZipFile('PennFudanPed.zip', 'r') as zip_ref:
        zip_ref.extractall()
    print("데이터셋 준비가 완료되었습니다.")
else:
    print("데이터셋이 이미 존재합니다.")
    
# 이미지 및 바운딩 박스 샘플 확인

# sample_img, sample_target = data_loader_test.dataset[0]
# sample_img = sample_img.permute(1, 2, 0).numpy()
# sample_boxes = sample_target['boxes'].numpy()

# plt.imshow(sample_img)
# for box in sample_boxes:
#     xmin, ymin, xmax, ymax = box
#     plt.gca().add_patch(
#         patches.Rectangle(
#             (xmin, ymin), xmax - xmin, ymax - ymin,
#             fill=False, edgecolor='g', linewidth=2
#         )
#     )
# plt.axis('off')  # 축 숨기기
# plt.show()        # 그래프 그리기


# # 앵커 박스 확인

# # 샘플 이미지 가져오기
# sample_img, _ = data_loader_test.dataset[0]
# sample_img_np = sample_img.permute(1, 2, 0).numpy()
# sample_img = sample_img.to(device)

# # 앵커 박스 생성
# imglist = ImageList(sample_img.unsqueeze(0), [(800, 800)])
# feature_map = backbone(sample_img.unsqueeze(0)).to(device)
# anchors = anchor_generator(imglist, [feature_map])

# # 샘플 이미지 출력
# fig, ax = plt.subplots(1, figsize=(12, 8))

# # 넓게 보기 위해 xlim과 ylim 설정
# margin = 400
# ax.set_xlim(-margin, 800 + margin)
# ax.set_ylim(800 + margin, -margin)
# ax.imshow(sample_img_np)

# # 앵커 박스 시각화
# for anchor in anchors:
#     for box in anchor:
#         box = box.cpu().numpy()
#         xmin, ymin, xmax, ymax = box
#         rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=0.1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

# plt.axis('off')
# plt.show()

num_classes = 2  # 배경 0 사람 1
model = FasterRCNN(backbone, rpn, roi_pooler, FasterRCNNHead(backbone.out_channels, num_classes), num_classes)
model.to(device)
model_save_path = 'fasterrcnn_model.pth'

# 랜덤 시드 고정
torch.manual_seed(0)
np.random.seed(0)

alpha = 0.5

FORCED_TRAIN = True
if not FORCED_TRAIN and os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("저장된 모델을 불러왔습지만 학습은 진행하지 않습니다.")

elif FORCED_TRAIN and os.path.exists(model_save_path):
    print("저장된 모델에 이어서 학습합니다.")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    best_test_loss = float('inf')
    best_params = None

    params = [p for p in model.parameters() if p.requires_grad]
    ##################################
    # 4. optimizer와 scheduler를 설정하세요.
    # 적절한 optimizer를 선택하고, lr과 weight_decay를 조절하세요.
    ##################################
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
    ##################################
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_loss = 0
        for images, targets in data_loader:

            images = [image.to(device) for image in images]
            images = torch.stack(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

            # if i % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")
            i += 1

        scheduler.step()
        print('======================================')
        print(f"Epoch [{epoch+1}] completed.")
        # 한 에폭마다 train loss, test loss 계산
        with torch.no_grad():

            test_loss = 0
            for images, targets in data_loader_test:
                images = [image.to(device) for image in images]
                images = torch.stack(images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()
            print(f"Train Loss: {total_loss / len(data_loader):.4f}")
            print(f"Test Loss: {test_loss / len(data_loader_test):.4f}")
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_params = copy.deepcopy(model.state_dict())
                print("Best model updated.")
        print('======================================')


    # 모델 저장
    model.load_state_dict(best_params)
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, model_save_path)
    print("모델을 저장하였습니다.")

else:
    print("저장된 모델이 없습니다. 학습을 시작합니다.")

    best_test_loss = float('inf')
    best_params = None

    params = [p for p in model.parameters() if p.requires_grad]
    ##################################
    # 4. optimizer와 scheduler를 설정하세요.
    # 적절한 optimizer를 선택하고, lr과 weight_decay를 조절하세요.
    ##################################
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
    ##################################
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_loss = 0
        for images, targets in data_loader:

            images = [image.to(device) for image in images]
            images = torch.stack(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

            # if i % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")
            i += 1

        scheduler.step()
        print('======================================')
        print(f"Epoch [{epoch+1}] completed.")
        # 한 에폭마다 train loss, test loss 계산
        with torch.no_grad():

            test_loss = 0
            for images, targets in data_loader_test:
                images = [image.to(device) for image in images]
                images = torch.stack(images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()
            print(f"Train Loss: {total_loss / len(data_loader):.4f}")
            print(f"Test Loss: {test_loss / len(data_loader_test):.4f}")
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_params = copy.deepcopy(model.state_dict())
                print("Best model updated.")
        print('======================================')


    # 모델 저장
    model.load_state_dict(best_params)
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, model_save_path)
    print("모델을 저장하였습니다.")
    
# test 데이터 추론 결과 시각화

try:
    print(f'Best_test_loss: {best_test_loss / len(data_loader_test):.4f}')
except:
    pass

model.eval()
i = 0
for images, targets in data_loader_test:
    images = [image.to(device) for image in images]
    images = torch.stack(images)
    # print(images)
    with torch.no_grad():
        predictions = model(images)
        # print(predictions)

    # 첫 번째 이미지와 예측 결과 가져오기
    img = images[0].permute(1, 2, 0).cpu().numpy()
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()

    # 예측된 바운딩 박스 그리기
    top_predictions = 3
    top_scores = sorted(pred_scores, reverse=True)[:top_predictions]

    if len(top_scores) == 0 or top_scores[0] < 0.000:
        continue

    # 원본 이미지 시각화
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        if score < top_scores[-1]:
            continue
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f'{score:.2f}', color='red', fontsize=12, weight='bold')

    # # 타겟 바운딩 박스 그리기 (옵션)
    # target_boxes = targets[0]['boxes'].cpu().numpy()
    # for box in target_boxes:
    #     xmin, ymin, xmax, ymax = box
    #     width, height = xmax - xmin, ymax - ymin
    #     rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')
    #     ax.add_patch(rect)

    plt.axis('off')
    plt.show()

    i += 1
    if i == 5:
        break
