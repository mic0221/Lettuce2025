#!/usr/bin/env python3
"""
Contains various utility functions

Adapted from Jacky To
"""
# Import functions
import os
import numpy as np
from pathlib import Path
from skimage import io
from PIL import Image
from torchinfo import summary
from shutil import copyfile
import json
import torch
import cv2


def save_img(img, target_dir, filename):
    """Saves np.ndarray or PIL.Image.Image image object to a file.

    Args:
        img (np.ndarray/PIL.Image.Image): Image object to be saved.
        target_dir (str): Target directory in which to save file.
        filename (str): Name of file to which the image is saved.

    Raises:
        Exception: Image should be either a np.ndarray or a PIL.Image.Image.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create filepath for writing the image file
    filepath = os.path.join(target_dir, filename)

    # Save img with skimage if np.ndarray
    if type(img) is np.ndarray:
        io.imsave(filepath, img, check_contrast=False)

    # Save img with PIL if Image
    elif type(img) is Image.Image:
        img.save(filepath)

    else:
        raise Exception("Input image object should be np.ndarray or PIL.Image.Image")


def binary_poly2px(filepath, custom_size=None):
    """Converts binary polygon mask in .json format from filepath to pixel mask.

    Args:
        filepath (str): Filepath of .json polygon mask.
        custom_size (list/tuple, optional): Contains height and width as int for custom mask size. Defaults to None.

    Returns:
        np.ndarray: Binary pixel mask of ints where 0 is background and 1 is the annotation.
    """
    # Open .json
    poly_json = json.load(open(filepath, "r"))

    # Create empty mask to draw on
    mask = np.zeros((poly_json["imageHeight"], poly_json["imageWidth"]))

    # Extract and draw points on empty mask
    for shape in poly_json["shapes"]:
        poly_points = shape["points"]
        poly_points = np.array(poly_points, dtype=np.int32)
        cv2.fillPoly(mask, [poly_points], color=1)

    if custom_size is not None and mask.shape[:2] != custom_size:
        mask = cv2.resize(mask, (custom_size[1], custom_size[0]), interpolation=cv2.INTER_NEAREST)

    return mask.astype(np.int32)

# Checkpointing
def save_checkpoint(state, target_dir, model_name):
    """Saves a PyTorch model state(s) to a target directory.

    Args:
        state (nn.Module): State(s) of a PyTorch model to save.
        target_dir (str): A directory for saving the model to.
        model_name (str): A filename for the saved model. Should include
        ".pth", ".pt", ".pth.tar" or "pt.tar" as the file extension.

    Example usage:
        save_model(state=model,
                    target_dir="models",
                    model_name="tipburn_resnet50_classifier.pth.tar")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(
        (".pth", ".pt", ".pth.tar", "pt.tar")
    ), "model_name should end with '.pth', '.pt', '.pth.tar' or 'pt.tar'"
    model_save_path = os.path.join(target_dir_path, model_name)

    # Save the model state_dict()
    torch.save(obj=state, f=model_save_path)
    print(f"[INFO] Saved model states to {model_save_path}")

def load_checkpoint(checkpoint, model):
    """Loads a saved PyTorch model state.

    Args:
        checkpoint (str): File of saved model state including path if necessary.
        model (nn.Module): PyTorch model on which the state will be applied.
    """
    checkpoint = torch.load(checkpoint, map_location="cpu") # remove map_location if not "cpu"
    model.load_state_dict(checkpoint["state_dict"])

    # model.load_state_dict(torch.load(checkpoint)["state_dict"])
    # print(f"[INFO] Loaded model state {checkpoint}")

# .json parser for config.json files
def parse_json(filepath):
    """Parses the a .json file as a dictionary containing all the values.

    Args:
        filepath (str): File path to the .json file.
    """
    # Parse .json files as dict with json's load function
    with open(filepath, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict

# Save training results, network summaries and config
def save_train_results(dict_results, target_dir, filename):
    """Writes training loop results to a tab-delimited text file.

    Input dictionary of results should be in format as constructed in train.py.
    Thus, with "epoch", "train_loss", "train_perform", "test_loss" and "test_perform"
    as keys.

    Args:
        dict_results (dict): Dictionary of training loop results.
        target_dir (str): Target directory in which to write the file.
        filename (str): File name of the text file to be written.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Get number of items to write rows
    num_items = len(dict_results["epoch"])

    # Write dictonary to file
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w") as f:
        # Check if multiple performance metrics were used
        if len(dict_results["train_perform"][0]) > 1:
            num_metrics = len(dict_results["train_perform"][0])
        else:
            num_metrics = 1

        # Write header
        headers = (
            ["epoch", "train_loss"]
            + ["train_perform"] * num_metrics
            + ["test_loss"]
            + ["test_perform"] * num_metrics
        )
        header_line = "\t".join(headers)
        f.write(header_line)
        f.write("\n")

        # Write rows
        for row in range(num_items):
            line_values = []
            perform_ind = 0
            for key in headers:
                # Writing each performance metric value
                if key in ["train_perform", "test_perform"]:
                    target_column = dict_results[key]
                    line_values.append(str(target_column[row][perform_ind]))
                    perform_ind += 1
                    if perform_ind == num_metrics:
                        perform_ind = 0

                # Writing the non-performance metric values
                else:
                    target_column = dict_results[key]
                    line_values.append(str(target_column[row]))
            line = "\t".join(line_values)
            f.write(line)
            f.write("\n")
    print(f"[INFO] Saved training results to {filepath}")

def save_network_summary(model, target_dir, filename, n_channels=3):
    """Writes a torchinfo summary and raw print summary of a model to a text file

    Args:
        model (nn.Module): PyTorch model to be summarized.
        target_dir (str): Target directory in which to write the file.
        filename (str): File name of the text file to be written.
        n_channels (int, optional): Number of input channels to test. Defaults to 3.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Get torchinfo summary of model
    model_stats = summary(
        model=model,
        input_size=(1, n_channels, 512, 512),
        col_names=["kernel_size", "input_size", "output_size", "num_params"],
        verbose=0,
    )
    str_model_stats = str(model_stats)

    # Get raw model as str (show all layers and parameters)
    str_raw_model = str(model)

    # Write model summaries to file
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w") as f:
        f.write("[TORCHINFO SUMMARY]")
        f.write("\n")
        f.write(str_model_stats)
        f.write("\n\n")
        f.write("[RAW SUMMARY]")
        f.write("\n")
        f.write(str_raw_model)
    print(f"[INFO] Saved network summaries to {filepath}")

def save_config(target_dir, filename, config_name="config.json"):
    """Saves config.json used for the model to a new text file.

    Args:
        target_dir (str): Target directory in which to write the file.
        filename (str): File name of the text file to be written.
        config_name (str): Name of config file to save. Defaults to "config.json".
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Copy config to new file
    filepath = os.path.join(target_dir, filename)
    copyfile(config_name, filepath)
    print(f"[INFO] Saved config.json to {filepath}")

# Performance metrics
def binary_jaccard(pred_logits, labels):
    """Calculate Jaccard index for binary semantic segmentation.

    When both predictions and ground-truths are empty, the Jaccard is considered 1.
    When predictions aren't empty  but the ground-truths are, the Jaccard is 0.
    If predictions and ground-truth aren't both empty, the Jaccard is simply the
    intersection over union.

    The Jaccard for a batch of images is calculated as the mean Jaccard index
    over the whole batch.

    Args:
        pred_logits (torch.Tensor): Batch of segmentation predictions as tensor of floats.
        labels (torch.Tensor): Batch of segmentation ground-truths as tensor of floats.

    Returns:
        float: Mean Jaccard index over a batch.
    """
    # Convert pred_logits to predictions of 0 or 1
    preds = torch.sigmoid(pred_logits)
    preds = (preds > 0.5).float()

    # Calculate Jaccards for every sample in batch
    jaccards = []
    for pred, label in zip(preds, labels):
        # If predictions and ground-truth are both empty
        if (pred.sum() == 0) and (label.sum() == 0):
            jaccard = 1

        # If predictions isn't empty but ground-truth is empty
        elif (pred.sum() > 0) and (label.sum() == 0):
            jaccard = 0

        # If predictions and ground-truth aren't both empty
        else:
            intersect = (pred * label).sum()
            union = pred.sum() + label.sum() - intersect
            jaccard = intersect / union

        if type(jaccard) == "torch.Tensor":
            jaccard = jaccard.item()
        jaccards.append(jaccard)

    # Calculate mean Jaccard over whole batch
    mean_jaccard = sum(jaccards) / len(jaccards)
    return mean_jaccard

def binary_Dice(pred_logits, labels):
    """Calculate Dice score for binary semantic segmentation.

    When both predictions and ground-truths are empty, the Jaccard is considered 1.
    When predictions aren't empty but the ground-truths are, the Jaccard is 0.
    If predictions and ground-truth aren't both empty, the Jaccard is simply the
    intersection over union.

    The Jaccard for a batch of images is calculated as the mean Jaccard index
    over the whole batch.

    Args:
        pred_logits (torch.Tensor): Batch of segmentation predictions as tensor of floats.
        labels (torch.Tensor): Batch of segmentation ground-truths as tensor of floats.

    Returns:
        float: Mean Jaccard index over a batch.
    """
    # Convert pred_logits to predictions of 0 or 1
    preds = torch.sigmoid(pred_logits)
    preds = (preds > 0.5).float()

    # Calculate Jaccards for every sample in batch
    dice = []
    for pred, label in zip(preds, labels):
        # If predictions and ground-truth are both empty
        if (pred.sum() == 0) and (label.sum() == 0):
            dice = 1

        # If predictions isn't empty but ground-truth is empty
        elif (pred.sum() > 0) and (label.sum() == 0):
            dice = 0

        # If predictions and ground-truth aren't both empty
        else:
            intersect = (pred * label).sum()
            total = pred.sum() + label.sum()
            dice = 2*intersect / total

        if type(dice) == "torch.Tensor":
            dice = dice.item()
        dice.append(dice)

    # Calculate mean Jaccard over whole batch
    mean_dice = sum(dice) / len(dice)
    return mean_dice


