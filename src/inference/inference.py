
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"






# Class names for the damage types
CLASS_NAMES = {
    # 0: "Severity Damage",       # Class 0
    1: "Minor Dent",            # Class 1
    2: "Minor Scratch",         # Class 2
    3: "Moderate Broken",       # Class 3
    4: "Moderate Dent",         # Class 4
    5: "Moderate Scratch",      # Class 5
    6: "Severe Broken",         # Class 6
    7: "Severe Dent",           # Class 7
    8: "Severe Scratch"         # Class 8
}

# Load the model
def load_model():
    logger.info("Loading the model...")
    cfg = get_cfg()
    cfg.merge_from_file("../model/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "../model/model_final.pth"  # Ensure this is the correct path to your weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # 9 classes
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
    logger.info("Model loaded successfully.")
    return cfg, DefaultPredictor(cfg)

# Function to run inference and visualize
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
    boxes = instances.pred_boxes.tensor.numpy().tolist()
    scores = instances.scores.numpy().tolist()
    labels = instances.pred_classes.numpy().tolist()

    # Log the results (showing both class number and class name)
    logger.info(f"Detected {len(boxes)} instances")
    for i, label in enumerate(labels):
        class_name = CLASS_NAMES.get(label, 'Unknown')
        logger.info(f"Class (Number): {label} - Class (Name): {class_name} - Score: {scores[i]:.2f}")

    # Set custom metadata for visualizer (fixing the 'Airplane' issue)
    metadata = MetadataCatalog.get("custom_dataset")
    metadata.set(thing_classes=[v for k, v in CLASS_NAMES.items()])

    # Visualize the result
    v = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert the visualization to PIL image and save it
    result_image = v.get_image()[:, :, ::-1]
    result_image_pil = Image.fromarray(result_image)
    result_image_pil.save("output_inference.jpg")
    logger.info("Inference complete. Visualized image saved as output_inference.jpg.")

if __name__ == "__main__":
    # Test the inference with an image
    test_image_path = r"D:\Assignment_lensor\vehicle_damage_detection\check\test.jpg"  # Replace with the path to an actual test image
    run_inference(test_image_path)
