# src/custom_roi_heads.py

from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from custom_loss import focal_loss
import torch
import torch.nn.functional as F


class CustomROIHeadsWithFocalLoss(StandardROIHeads):
    """
    Custom ROI heads to implement focal loss for classification.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # Initialize the ROI pooler (pooler takes features and boxes as input and produces fixed-size outputs)
        self.pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_HEADS.POOLER_RESOLUTION,
            scales=[1.0 / input_shape[f].stride for f in self.in_features],
            sampling_ratio=cfg.MODEL.ROI_HEADS.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_HEADS.POOLER_TYPE,
        )

    def forward(self, images, features, proposals, gt_instances):
        """
        Override the forward method to apply Focal Loss with ignore_index.
        Args:
            - images: Image tensors
            - features: Extracted feature maps
            - proposals: Proposals generated from the RPN
            - gt_instances: Ground truth instances (with gt_classes)
        """
        # Call the parent class's forward method to get the feature maps
        features_list = [features[f] for f in self.in_features]  # Extract features from the input feature map

        # Apply ROI Pooling on the features to get pooled features for the box head
        pooled_features = self.pooler(features_list, [x.proposal_boxes for x in proposals])

        # Flatten pooled features into [batch_size * num_rois, -1]
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # The ROI head now has pooled features that can be passed to the box predictor
        logits, bbox_deltas = self.box_predictor(
            pooled_features)  # This will give us classification logits and bounding box deltas

        # Get the device from logits (which contain the classification scores)
        device = logits.device

        # Move ground truth class labels (gt_classes) to the same device as the classification logits
        gt_classes = [inst.gt_classes.to(device) for inst in gt_instances]

        # Concatenate ground truth classes into a single tensor
        gt_classes_tensor = torch.cat(gt_classes).to(device)

        # Apply Focal Loss between the classification logits and ground truth labels
        loss_cls = focal_loss(logits, gt_classes_tensor, alpha=0.25, gamma=2.0, ignore_index=0)

        # Add the new classification loss into the detector losses
        detector_losses = {"loss_cls": loss_cls}

        return proposals, detector_losses
