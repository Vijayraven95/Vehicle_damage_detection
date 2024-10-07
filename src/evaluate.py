# src/evaluate.py

import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from config import setup_config
from train import register_datasets


def evaluate():
    output_dir = "./output"
    num_classes = 7  # 7 damage classes (excluding 'severity-damage')

    # Register the datasets
    register_datasets()

    # Load the trained model for evaluation
    cfg = setup_config(output_dir=output_dir, num_classes=num_classes)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust if needed

    # Set up the evaluator
    evaluator = COCOEvaluator("vehicle_train", cfg, False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, "vehicle_train")

    # Perform evaluation
    trainer = DefaultTrainer(cfg)  # Load trainer
    results = inference_on_dataset(trainer.model, test_loader, evaluator)

    # Print evaluation results
    print("Evaluation Results:", results)


if __name__ == "__main__":
    evaluate()
