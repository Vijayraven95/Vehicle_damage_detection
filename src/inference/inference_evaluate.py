import logging
import numpy as np
import json
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from PIL import Image
import cv2
from detectron2.data.datasets import register_coco_instances

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Class names for the damage types (background class 0 removed)
CLASS_NAMES = {
    1: "minor-dent",  # Class 1
    2: "minor-scratch",  # Class 2
    3: "moderate-broken",  # Class 3
    4: "moderate-dent",  # Class 4
    5: "moderate-scratch",  # Class 5
    6: "severe-broken",  # Class 6
    7: "severe-dent",  # Class 7
    8: "severe-scratch"  # Class 8
}

# Load the model
def load_model():
    logger.info("Loading the model...")
    cfg = get_cfg()
    cfg.merge_from_file("../config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "../output/model_final.pth"  # Path to your trained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Update to 8 classes
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
    logger.info("Model loaded successfully.")
    return cfg, DefaultPredictor(cfg)


# Function to run inference and visualize for a single image
def run_inference(image_path):
    cfg, predictor = load_model()

    # Open the image
    if not os.path.exists(image_path):
        logger.error(f"Image {image_path} does not exist.")
        return

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)[:, :, ::-1].copy()  # Convert to BGR for model inference

    # Run inference
    logger.info(f"Running inference on image {image_path}...")
    outputs = predictor(image_np)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        logger.info(f"No instances detected for image {image_path}")
        return

    boxes = instances.pred_boxes.tensor.numpy().tolist()
    scores = instances.scores.numpy().tolist()
    labels = instances.pred_classes.numpy().tolist()

    # Log the results (showing both class number and class name)
    logger.info(f"Detected {len(boxes)} instances")
    for i, label in enumerate(labels):
        class_name = CLASS_NAMES.get(label, 'Unknown')
        logger.info(f"Class (Number): {label} - Class (Name): {class_name} - Score: {scores[i]:.2f}")

    # Fetch metadata
    metadata = MetadataCatalog.get("vehicle_test")

    # Visualize the result
    v = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(instances)

    # Convert the visualization to PIL image and save it
    result_image = v.get_image()[:, :, ::-1]
    result_image_pil = Image.fromarray(result_image)
    result_image_pil.save("output_inference.jpg")
    logger.info("Inference complete. Visualized image saved as output_inference.jpg.")


# Function to run inference and evaluate the entire test set
def evaluate_testset(test_dataset_name, output_json="test_results.json"):
    cfg, predictor = load_model()

    # Load the test dataset
    logger.info(f"Loading test dataset: {test_dataset_name}")
    test_metadata = MetadataCatalog.get(test_dataset_name)
    test_dataset = DatasetCatalog.get(test_dataset_name)

    # Prepare evaluator for COCO-style evaluation
    evaluator = COCOEvaluator(test_dataset_name, cfg, False, output_dir="output/")

    # Build the test loader
    test_loader = build_detection_test_loader(cfg, test_dataset_name)

    # Run inference on the test dataset and compare predictions with annotations
    logger.info(f"Running inference and evaluation on the test dataset {test_dataset_name}...")
    evaluation_results = inference_on_dataset(predictor.model, test_loader, evaluator)

    logger.info("Evaluation complete. Metrics: ")
    logger.info(evaluation_results)

    # Save the evaluation results to a JSON file
    with open(output_json, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    logger.info(f"Evaluation results saved to {output_json}")


if __name__ == "__main__":
    # Register the test dataset
    register_coco_instances(
        "vehicle_test",  # Name for your test dataset (used in evaluate_testset)
        {},
        r"/data/annotations/instances_test.json",  # Path to your COCO-style annotations file (JSON)
        r"D:\Assignment_lensor\vehicle_damage_detection\data\test"  # Directory containing test images
    )

    # Test single image inference
    test_image_path = r"D:\Assignment_lensor\vehicle_damage_detection\check\test.jpg"  # Replace with the path to an actual test image
    run_inference(test_image_path)

    # Evaluate the entire test set
    test_dataset_name = "vehicle_test"  # Name of the dataset registered
    evaluate_testset(test_dataset_name)
