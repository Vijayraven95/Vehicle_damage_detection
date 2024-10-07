import logging
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image
import numpy as np
from io import BytesIO
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Instantiate FastAPI app
app = FastAPI()

# Class names for the damage types
CLASS_NAMES = {
    1: "Minor Dent",
    2: "Minor Scratch",
    3: "Moderate Broken",
    4: "Moderate Dent",
    5: "Moderate Scratch",
    6: "Severe Broken",
    7: "Severe Dent",
    8: "Severe Scratch"
}


# Load the model
def load_model():
    logger.info("Loading the model...")

    # Get the directory path where main.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths relative to the VDD folder (main.py is in the app folder)
    config_path = os.path.join(base_dir, "model/mask_rcnn_R_50_FPN_3x.yaml")
    model_weights_path = os.path.join(base_dir, "model/model_final.pth")

    # Check if the config and model files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)  # Load config.yaml
    cfg.MODEL.WEIGHTS = model_weights_path  # Load model checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for predictions
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Number of classes (8 in this case)

    # Use GPU if available, otherwise fallback to CPU
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}.")
    return cfg, DefaultPredictor(cfg)


# Load model once at app startup
cfg, predictor = load_model()


# FastAPI route to handle image upload and run inference
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image data from the uploaded file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Resize the image to reduce processing time (e.g., to 800x800)
        image = image.resize((800, 800))

        image_np = np.array(image)[:, :, ::-1].copy()  # Convert to BGR for model inference

        # Run inference
        logger.info(f"Running inference on uploaded image...")
        outputs = predictor(image_np)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        labels = instances.pred_classes.numpy().tolist()

        # Log the results
        logger.info(f"Detected {len(boxes)} instances")
        for i, label in enumerate(labels):
            class_name = CLASS_NAMES.get(label, 'Unknown')
            logger.info(f"Class (Number): {label} - Class (Name): {class_name} - Score: {scores[i]:.2f}")

        # Return predictions as JSON
        return {"boxes": boxes, "scores": scores, "labels": labels}

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"error": str(e)}


# FastAPI route to predict and visualize the result
@app.post("/predict_and_visualize")
async def predict_and_visualize(file: UploadFile = File(...)):
    try:
        # Read image data from the uploaded file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Resize the image to reduce processing time (e.g., to 800x800)
        image = image.resize((800, 800))

        image_np = np.array(image)[:, :, ::-1].copy()  # Convert to BGR for model inference

        # Run inference
        logger.info(f"Running inference and visualization...")
        outputs = predictor(image_np)

        # Set custom metadata for visualizer
        metadata = MetadataCatalog.get("custom_dataset")
        metadata.set(thing_classes=[v for k, v in CLASS_NAMES.items()])

        # Visualize the result
        v = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Convert the visualization to a PIL image
        result_image = v.get_image()[:, :, ::-1]
        result_image_pil = Image.fromarray(result_image)

        # Return the visualized image as a StreamingResponse
        img_byte_arr = BytesIO()
        result_image_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        return {"error": str(e)}
