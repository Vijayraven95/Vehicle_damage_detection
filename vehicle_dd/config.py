# config.py

from detectron2.config import get_cfg

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("Base-RCNN-FPN.yaml")  # Path to your config file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for prediction
    cfg.MODEL.DEVICE = "cpu"  # Using CPU for inference
    cfg.MODEL.WEIGHTS = "./model/model_final.pth"  # Path to the trained model weights
    return cfg
