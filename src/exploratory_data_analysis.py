import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import os

# Set this environment variable to bypass the OpenMP conflict issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths to the datasets
TRAIN_PATH = "../data/train"
TRAIN_ANNOTATIONS = "../data/annotations/instances_train.json"
VAL_PATH = "../data/val"
VAL_ANNOTATIONS = "../data/annotations/instances_val.json"
TEST_PATH = "../data/test"
TEST_ANNOTATIONS = "../data/annotations/instances_test.json"

# Class names for the damage types (assuming 9 classes)
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

# Create output folder for results
OUTPUT_DIR = "exploratory_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def register_datasets():
    """
    Register the COCO datasets for Detectron2 for train, validation, and test.
    """
    register_coco_instances("vehicle_train", {}, TRAIN_ANNOTATIONS, TRAIN_PATH)
    register_coco_instances("vehicle_val", {}, VAL_ANNOTATIONS, VAL_PATH)
    register_coco_instances("vehicle_test", {}, TEST_ANNOTATIONS, TEST_PATH)


# Helper function to plot bar graphs and save
def plot_bar(data, x_label, y_label, title, filename):
    """
    Plot a bar graph for given data and save the result.

    Args:
        data: Dictionary containing the data for the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Title of the plot.
        filename: Name of the file to save the plot as.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(data.keys()), y=list(data.values()))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def class_distribution(dataset, metadata):
    """
    Analyze and plot the number of instances per class in the dataset.

    Args:
        dataset: List of dataset samples with annotations.
        metadata: Metadata object for the dataset.
    """
    category_names = metadata.thing_classes
    class_count = {category: 0 for category in category_names}

    for item in dataset:
        annotations = item['annotations']
        for annotation in annotations:
            class_count[category_names[annotation['category_id']]] += 1

    plot_bar(class_count, 'Class Names', 'Instance Count', 'Class Distribution in the Dataset', 'class_distribution.png')


def bbox_size_distribution(dataset):
    """
    Analyze and plot the bounding box size distribution in the dataset.

    Args:
        dataset: List of dataset samples with annotations.
    """
    bbox_sizes = []

    for item in dataset:
        annotations = item['annotations']
        for annotation in annotations:
            bbox = annotation['bbox']
            width, height = bbox[2], bbox[3]
            area = width * height
            bbox_sizes.append(area)

    plt.figure(figsize=(10, 6))
    sns.histplot(bbox_sizes, bins=50, kde=True)
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Frequency')
    plt.title('Bounding Box Size Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_size_distribution.png'))
    plt.close()


def instances_per_image(dataset):
    """
    Analyze and plot the number of instances per image in the dataset.

    Args:
        dataset: List of dataset samples with annotations.
    """
    instance_counts = []

    for item in dataset:
        annotations = item['annotations']
        instance_counts.append(len(annotations))

    plt.figure(figsize=(10, 6))
    sns.histplot(instance_counts, bins=20, kde=False)
    plt.xlabel('Number of Instances per Image')
    plt.ylabel('Frequency')
    plt.title('Distribution of Instances per Image')
    plt.savefig(os.path.join(OUTPUT_DIR, 'instances_per_image.png'))
    plt.close()


def visualize_sample_images(dataset, metadata, num_samples=5):
    """
    Visualize a random set of sample images with annotations and save the results.

    Args:
        dataset: List of dataset samples with annotations.
        metadata: Metadata object for the dataset.
        num_samples: Number of samples to visualize.
    """
    for idx, d in enumerate(random.sample(dataset, num_samples)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(10, 6))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, f'sample_image_{idx}.png'))
        plt.close()


def visualize_per_class(dataset, metadata, class_names):
    """
    Visualize one sample image per class with annotations and save the results.

    Args:
        dataset: List of dataset samples with annotations.
        metadata: Metadata object for the dataset.
        class_names: Dictionary of class names.
    """
    samples_per_class = {class_id: None for class_id in class_names.keys()}

    for item in dataset:
        annotations = item['annotations']
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id in samples_per_class and samples_per_class[class_id] is None:
                samples_per_class[class_id] = item
            if all(v is not None for v in samples_per_class.values()):
                break

    num_classes = len(class_names)
    fig, axs = plt.subplots(1, num_classes, figsize=(20, 5))

    for i, (class_id, sample) in enumerate(samples_per_class.items()):
        if sample is not None:
            img = cv2.imread(sample["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(sample)

            axs[i].imshow(vis.get_image()[:, :, ::-1])
            axs[i].set_title(class_names[class_id])
            axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'visualize_per_class.png'))
    plt.close()


def class_co_occurrence_matrix(dataset, class_names):
    """
    Compute and visualize the co-occurrence matrix of classes in the dataset.

    Args:
        dataset: List of dataset samples with annotations.
        class_names: Dictionary of class names.
    """
    co_occurrence_matrix = np.zeros((len(class_names), len(class_names)))

    for item in dataset:
        present_classes = set()
        annotations = item['annotations']
        for annotation in annotations:
            present_classes.add(annotation['category_id'] - 1)  # 0-indexed classes

        for cls1 in present_classes:
            for cls2 in present_classes:
                co_occurrence_matrix[cls1, cls2] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence_matrix, annot=True, cmap="Blues", xticklabels=class_names.values(),
                yticklabels=class_names.values())
    plt.title("Class Co-Occurrence Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_co_occurrence_matrix.png'))
    plt.close()


def bbox_aspect_ratio_distribution(dataset):
    """
    Analyze and plot the aspect ratio (width/height) of bounding boxes in the dataset.

    Args:
        dataset: List of dataset samples with annotations.
    """
    aspect_ratios = []

    for item in dataset:
        annotations = item['annotations']
        for annotation in annotations:
            bbox = annotation['bbox']
            width, height = bbox[2], bbox[3]
            aspect_ratio = width / height if height > 0 else 0
            aspect_ratios.append(aspect_ratio)

    plt.figure(figsize=(10, 6))
    sns.histplot(aspect_ratios, bins=50, kde=True)
    plt.xlabel('Aspect Ratio (Width/Height)')
    plt.ylabel('Frequency')
    plt.title('Bounding Box Aspect Ratio Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_aspect_ratio_distribution.png'))
    plt.close()


# Main function to perform EDA
if __name__ == "__main__":
    # Register the datasets
    register_datasets()

    # Load the dataset for analysis
    train_metadata = MetadataCatalog.get("vehicle_train")
    train_dataset = DatasetCatalog.get("vehicle_train")

    # Perform EDA on the training dataset
    class_distribution(train_dataset, train_metadata)
    bbox_size_distribution(train_dataset)
    instances_per_image(train_dataset)
    visualize_sample_images(train_dataset, train_metadata)
    visualize_per_class(train_dataset, train_metadata, CLASS_NAMES)
    bbox_aspect_ratio_distribution(train_dataset)
    class_co_occurrence_matrix(train_dataset, CLASS_NAMES)
