# src/train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.events import TensorboardXWriter
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from config import setup_config
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, EventStorage

# Set the CUDA_LAUNCH_BLOCKING environment variable for debugging (optional)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
setup_logger()

# Function to register COCO datasets (train, val, and test)
def register_datasets():
    register_coco_instances("vehicle_train", {}, "../data/annotations/instances_train.json", "../data/train")
    register_coco_instances("vehicle_val", {}, "../data/annotations/instances_val.json", "../data/val")
    register_coco_instances("vehicle_test", {}, "../data/annotations/instances_test.json", "../data/test")

# Function to ensure output directory exists with correct permissions
def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to the output directory: {output_dir}")

# Custom Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ignore_mask = (targets != self.ignore_index)
        log_p = inputs.log()
        p_t = inputs
        loss = -self.alpha * ((1 - p_t) ** self.gamma) * log_p
        loss = loss * ignore_mask.float()
        return loss.mean()

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        model.roi_heads.box_predictor.loss_func = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=0)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_writers(self):
        """
        Override the default build_writers to ensure Tensorboard logging along with other loggers.
        """
        # Set up writers: CommonMetricPrinter (console), JSONWriter (metrics.json), and TensorboardXWriter
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),  # Console logging
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),  # JSON file for metrics
            TensorboardXWriter(self.cfg.OUTPUT_DIR),  # TensorboardXWriter for TensorBoard logging
        ]

    def run_validation(self):
        evaluator = COCOEvaluator("vehicle_val", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, "vehicle_val")
        results = inference_on_dataset(self.model, val_loader, evaluator)
        return results

def main():
    output_dir = "./output"
    num_classes = 8

    register_datasets()

    cfg = setup_config(output_dir=output_dir, num_classes=num_classes)
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATASETS.TRAIN = ("vehicle_train",)
    cfg.DATASETS.TEST = ("vehicle_val",)  # Validation dataset
    cfg.TEST.EVAL_PERIOD = 2000  # Every 1000 iterations

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
