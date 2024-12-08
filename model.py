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

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import RoIAlign

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############################
# 1. RPN 앵커의 크기 및 종횡비를 설정하세요
##############################
anchor_generator = AnchorGenerator(
    sizes=((128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
###############################

# ResNet50 백본 사용
backbone = models.resnet50(pretrained=True).to(device)

# 마지막 분류 계층 제거
modules = list(backbone.children())[:-2]
backbone = nn.Sequential(*modules)

backbone.out_channels = 2048  # ResNet50의 마지막 특징 맵 채널 수

# RPN, RoI Pooler 생성


rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

rpn = RegionProposalNetwork(
    anchor_generator,
    rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_top_n=dict(training=2000, testing=1000),
    post_nms_top_n=dict(training=2000, testing=1000),
    nms_thresh=0.7
)


roi_pooler = RoIAlign(
    output_size=(7, 7),
    ###################
    # 2. spatial_scale을 설정하세요.
    # 백본의 feature map 크기와 output_size를 고려하여 설정하세요.
    ###################
    spatial_scale=1/32,
    ###################
    sampling_ratio=2
)

# FasterRCNNHead 생성

################ 성능 향상을 위해 조절하기 과제가 가능할 것 같습니다. ################
class FasterRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FasterRCNNHead, self).__init__()
        #############################################
        # 3. fc1, fc2, classifier, box_regressor를 forward를 참고해 구현하세요.
        # classifier는 num_classes만큼의 클래스를 출력해야 합니다.
        # box_regressor는 (num_classes-1) * 4만큼의 값을 출력해야 합니다.
        #############################################
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.box_regressor = nn.Linear(1024, (num_classes-1) * 4)
        #############################################

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        scores = self.classifier(x)
        bbox_deltas = self.box_regressor(x)
        return scores, bbox_deltas



# FasterRCNN 모델 생성

from torchvision.ops.boxes import box_iou

class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_pooler, head, num_classes,
                 min_size=800, max_size=1333,
                 score_thresh=0.05, nms_thresh=0.5, detections_per_img=100):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone

        # backbone train 여부 조절
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.rpn = rpn
        self.roi_pooler = roi_pooler
        self.head = head
        self.num_classes = num_classes

        # 후처리에 필요한 설정값들
        self.score_thresh = score_thresh  # 점수 임계값
        self.nms_thresh = nms_thresh      # NMS 임계값
        self.detections_per_img = detections_per_img  # 이미지당 최대 검출 수

        self.min_size = min_size
        self.max_size = max_size

    def transform(self, images):
        # 이미지 크기 조정 및 정규화 (필요한 경우)
        image_sizes = [img.shape[-2:] for img in images]
        images = [img for img in images]
        images = torch.stack(images)
        return ImageList(images, image_sizes)

    def forward(self, images, targets=None):
        # 이미지 전처리
        if self.training:
            assert targets is not None
        images = self.transform(images)

        # 특징 추출
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # RPN 단계
        proposals, rpn_losses = self.rpn(images, features, targets)

        # RoI 풀링
        box_features = self.roi_pooler(features['0'], proposals)

        # 분류 및 회귀 헤드
        class_logits, box_regression = self.head(box_features)
        box_regression = box_regression.view(-1, self.num_classes-1, 4)

        result = []
        losses = {}

        if self.training:
            # 학습 시 손실 계산
            detector_losses = self.compute_loss(class_logits, box_regression, targets, proposals)
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses
        else:
            # 추론 시 결과 반환
            detections = self.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            return detections

    def compute_loss(self, class_logits, box_regression, targets, proposals):
        # 각 이미지에 대해 타겟 할당 및 손실 계산
        labels, regression_targets = self.assign_targets_to_proposals(proposals, targets)

        # 분류 손실 계산
        loss_classifier = F.cross_entropy(class_logits, labels)

        # 박스 회귀 손실 계산
        box_regression = box_regression.view(-1, self.num_classes-1, 4)

        # 배경이 아닌 인덱스 선택
        foreground_indices = labels > 0
        labels_pos = labels[foreground_indices] - 1  # 클래스 인덱스 조정 (1부터 시작하므로 1 빼기)

        # 유효한 예측과 타겟 추출
        box_regression = box_regression[foreground_indices, labels_pos]
        regression_targets = regression_targets[foreground_indices]

        # Smooth L1 손실 계산
        loss_box_reg = F.smooth_l1_loss(
            box_regression, regression_targets, beta=1/9, reduction='sum') / labels.numel()

        return {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg}

    def assign_targets_to_proposals(self, proposals, targets):
        labels = []
        matched_gt_boxes = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            device = proposals_per_image.device
            gt_boxes = targets_per_image["boxes"].to(device)
            gt_labels = targets_per_image["labels"].to(device)

            if gt_boxes.numel() == 0:
                # 타겟이 없는 경우 모든 proposals를 배경으로 처리
                labels_per_image = torch.zeros((proposals_per_image.shape[0],), dtype=torch.int64, device=device)
                matched_gt_boxes_per_image = torch.zeros_like(proposals_per_image)
            else:
                # IoU 계산
                ious = box_iou(proposals_per_image, gt_boxes)

                # 각 proposal에 대해 최대 IoU를 가지는 gt 박스의 인덱스와 IoU 값 얻기
                max_iou_values, gt_assignment = ious.max(dim=1)

                # 라벨 초기화 (배경은 0)
                labels_per_image = torch.zeros((proposals_per_image.shape[0],), dtype=torch.int64, device=device)

                # 포그라운드와 배경 구분을 위한 임계값 설정
                foreground_idxs = max_iou_values >= 0.5
                background_idxs = max_iou_values < 0.5

                # 포그라운드인 proposals에 gt 라벨 할당
                labels_per_image[foreground_idxs] = gt_labels[gt_assignment[foreground_idxs]]

                # 매칭된 gt 박스 저장
                matched_gt_boxes_per_image = gt_boxes[gt_assignment]

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        labels = torch.cat(labels, dim=0)
        regression_targets = self.box_coder_encode(torch.cat(proposals, dim=0), torch.cat(matched_gt_boxes, dim=0))

        return labels, regression_targets


    def get_regression_targets(self, proposals, targets):
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if targets_per_image["boxes"].numel() == 0:
                # 타겟이 없는 경우 회귀 타겟은 0으로 설정
                regression_targets_per_image = torch.zeros_like(proposals_per_image)
            else:
                # 각 proposal에 해당하는 타겟 박스를 가져옴
                target_boxes = targets_per_image["boxes"][0].expand(proposals_per_image.shape[0], 4)
                regression_targets_per_image = self.box_coder_encode(proposals_per_image, target_boxes)

            regression_targets.append(regression_targets_per_image)

        regression_targets = torch.cat(regression_targets, dim=0)
        return regression_targets
    def box_coder_encode(self, proposals, targets):
        # 박스 코딩 (targets - proposals)
        wx, wy, ww, wh = (1.0, 1.0, 1.0, 1.0)
        proposals = proposals.float()
        targets = targets.float()

        px = (proposals[:, 0] + proposals[:, 2]) / 2
        py = (proposals[:, 1] + proposals[:, 3]) / 2
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]

        gx = (targets[:, 0] + targets[:, 2]) / 2
        gy = (targets[:, 1] + targets[:, 3]) / 2
        gw = targets[:, 2] - targets[:, 0]
        gh = targets[:, 3] - targets[:, 1]

        dx = wx * (gx - px) / pw
        dy = wy * (gy - py) / ph
        dw = ww * torch.log(gw / pw)
        dh = wh * torch.log(gh / ph)

        return torch.stack((dx, dy, dw, dh), dim=1)

    def postprocess_detections(self, class_logits, box_regression, proposals, image_sizes):
        device = class_logits.device
        num_classes = self.num_classes

        # 분류 점수 계산
        pred_scores = F.softmax(class_logits, -1)

        # 이미지별로 결과 분리
        boxes_per_image = [len(p) for p in proposals]
        pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
        box_regression_list = box_regression.split(boxes_per_image, dim=0)
        results = []

        for scores, box_regression_per_image, proposals_per_image, image_size in zip(pred_scores_list, box_regression_list, proposals, image_sizes):
            if isinstance(proposals_per_image, list):
                # 리스트를 텐서로 변환
                if len(proposals_per_image) == 0:
                    # proposals_per_image가 비어있는 경우 빈 텐서를 생성
                    proposals_per_image = torch.empty((0, 4), device=device)
                else:
                    proposals_per_image = torch.stack(proposals_per_image)
            else:
                proposals_per_image = proposals_per_image.to(device)

            # 박스 디코딩
            pred_boxes = self.box_coder_decode(box_regression_per_image, proposals_per_image)
            # 결과 필터링
            boxes, labels, scores = self.filter_results(pred_boxes, scores, image_size)
            results.append({'boxes': boxes, 'labels': labels, 'scores': scores})
        return results


    def box_coder_decode(self, rel_codes, boxes):
        # boxes가 텐서인지 확인하고, 아니면 텐서로 변환
        if isinstance(boxes, list):
            boxes = torch.stack(boxes)
        boxes = boxes.to(rel_codes.dtype)
        total_boxes = boxes.size(0)
        num_classes = self.num_classes-1

        rel_codes = rel_codes.view(total_boxes, num_classes, 4)
        boxes = boxes[:, None, :].expand(total_boxes, num_classes, 4)

        # 이후 코드 동일
        px = (boxes[..., 0] + boxes[..., 2]) * 0.5
        py = (boxes[..., 1] + boxes[..., 3]) * 0.5
        pw = boxes[..., 2] - boxes[..., 0]
        ph = boxes[..., 3] - boxes[..., 1]

        dx = rel_codes[..., 0]
        dy = rel_codes[..., 1]
        dw = rel_codes[..., 2]
        dh = rel_codes[..., 3]

        gx = dx * pw + px
        gy = dy * ph + py
        gw = pw * torch.exp(dw)
        gh = ph * torch.exp(dh)

        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5

        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return decoded_boxes


    def filter_results(self, boxes, scores, image_shape):
        num_classes = scores.shape[1]  # 배경 클래스를 포함한 클래스 수
        device = scores.device

        # 결과를 저장할 리스트 초기화
        final_boxes = []
        final_scores = []
        final_labels = []

        # 각 클래스에 대해 반복 (배경 클래스는 인덱스 0이므로 건너뜀)
        for cls_ind in range(1, num_classes):
            cls_scores = scores[:, cls_ind]
            cls_boxes = boxes  # 박스는 모든 클래스에서 공유
            cls_boxes = cls_boxes.view(-1, 4)
            # 점수 임계값 적용
            score_keep = cls_scores > self.score_thresh
            cls_scores = cls_scores[score_keep]
            cls_boxes = cls_boxes[score_keep]


            if cls_boxes.numel() == 0:
                continue  # 남은 박스가 없으면 다음 클래스로

            # NMS 적용
            nms_indices = nms(cls_boxes, cls_scores, self.nms_thresh)
            nms_indices = nms_indices[:self.detections_per_img]

            cls_scores = cls_scores[nms_indices]
            cls_boxes = cls_boxes[nms_indices]

            # 레이블 할당 (현재 클래스)
            cls_labels = torch.full((len(nms_indices),), cls_ind, dtype=torch.int64, device=device)

            final_boxes.append(cls_boxes)
            final_scores.append(cls_scores)
            final_labels.append(cls_labels)

        if len(final_boxes) == 0:
            # 검출된 결과가 없으면 빈 텐서 반환
            return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), dtype=torch.int64, device=device)

        # 모든 클래스의 결과를 연결
        boxes = torch.cat(final_boxes, dim=0)
        scores = torch.cat(final_scores, dim=0)
        labels = torch.cat(final_labels, dim=0)

        # 검출 수 제한
        if boxes.numel() > 0:
            scores, order = scores.sort(descending=True)
            order = order[:self.detections_per_img]
            boxes = boxes[order]
            labels = labels[order]
            scores = scores[:self.detections_per_img]

        # 박스 좌표를 이미지 크기에 맞게 조정
        boxes[:, 0].clamp_(min=0, max=image_shape[1])  # x1
        boxes[:, 1].clamp_(min=0, max=image_shape[0])  # y1
        boxes[:, 2].clamp_(min=0, max=image_shape[1])  # x2
        boxes[:, 3].clamp_(min=0, max=image_shape[0])  # y2

        return boxes, labels, scores
    
    