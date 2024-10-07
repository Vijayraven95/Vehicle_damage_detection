# src/config.py

from detectron2.config import get_cfg
import os

def setup_config(output_dir, num_classes=8, train_dataset="vehicle_train", val_dataset="vehicle_val", batch_size=2,
                 max_iter=50000):
    """
    Sets up the configuration for training Mask R-CNN model in Detectron2.

    :param output_dir: Directory where outputs (model, logs) will be saved
    :param num_classes: Number of classes (excluding background)
    :param train_dataset: Name of the training dataset registered in Detectron2
    :param val_dataset: Name of the validation dataset registered in Detectron2
    :param batch_size: Batch size during training
    :param max_iter: Number of training iterations
    :return: Config object
    """
    cfg = get_cfg()
    # cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(r"D:\Assignment_lensor\vehicle_damage_detection\src\config\mask_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Use the pre-trained weights from COCO for transfer learning
    cfg.MODEL.WEIGHTS = "./pretrained_weights/model_final_f10217.pkl"

    # Number of classes (8 because 'severity-damage' is excluded)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Training hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # Learning rate decay is not used by default
    # Set to CPU for debugging, use 'cuda' for GPU training
    cfg.MODEL.DEVICE = "cpu"  # Use GPU, set to "cpu" for CPU
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster RoI head training
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold during inference
    # # Ensure that POOLER parameters exist
    # cfg.MODEL.ROI_HEADS.POOLER_RESOLUTION = 7  # Set default 7x7
    # cfg.MODEL.ROI_HEADS.POOLER_SAMPLING_RATIO = 2  # Set default sampling ratio
    # cfg.MODEL.ROI_HEADS.POOLER_TYPE = "ROIAlign"  # Set default pooler type
    return cfg
