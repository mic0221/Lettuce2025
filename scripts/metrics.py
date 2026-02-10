#!/usr/bin/env python3
"""
Evaluation metrics for model performance,
including accuracy, jaccard index, dice score, precision, recall.

patch: an option to evaluate on patch-based model (True) or resize-based model (False)

Adapted from Jacky To
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

# Performance metrics
def binary_jaccard(pred_masks, labels):
    """Calculate Jaccard index for binary semantic segmentation.

    When both predictions and ground-truths are empty, the Jaccard is considered 1.
    When predictions aren't empty  but the ground-truths are, the Jaccard is 0.
    If predictions and ground-truth aren't both empty, the Jaccard is simply the
    intersection over union.

    Args:
        pred_masks (numpy.ndarray): Predicted binary segmentation masks.
        labels (numpy.ndarray): Ground-truth binary segmentation masks.

    Returns:
        jaccard (float): Mean Jaccard index over a batch.
    """
    # If predictions and ground-truth are both empty
    if (pred_masks.sum() == 0) and (labels.sum() == 0):
        jaccard = 1

    # If predictions isn't empty but ground-truth is empty
    elif (pred_masks.sum() > 0) and (labels.sum() == 0):
        jaccard = 0

    # If predictions and ground-truth aren't both empty
    else:
        intersect = (pred_masks * labels).sum()
        union = pred_masks.sum() + labels.sum() - intersect
        jaccard = intersect / union

    return jaccard

def binary_Dice(pred_masks, labels):
    """Calculate Dice score for binary semantic segmentation.

    When both predictions and ground-truths are empty, the Jaccard is considered 1.
    When predictions aren't empty but the ground-truths are, the Jaccard is 0.
    If predictions and ground-truth aren't both empty, the Jaccard is simply the
    intersection over union.


    Args:
        pred_masks (numpy.ndarray): Predicted binary segmentation masks.
        labels (numpy.ndarray): Ground-truth binary segmentation masks.

    Returns:
        dice (float): Mean dice index over a batch.
    """

    # If predictions and ground-truth are both empty
    if (pred_masks.sum() == 0) and (labels.sum() == 0):
        dice = 1

    # If predictions isn't empty but ground-truth is empty
    elif (pred_masks.sum() > 0) and (labels.sum() == 0):
        dice = 0

    # If predictions and ground-truth aren't both empty
    else:
        intersect = (pred_masks * labels).sum()
        total = pred_masks.sum() + labels.sum()
        dice = 2*intersect / total

    return dice

def binary_all(pred_masks, labels):
    """Compute pixel-level confusion matrix components for binary segmentation.

    This function calculates the total number of
    true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN)
    by comparing predicted binary masks with ground-truth labels.

    Args:
        pred_masks (numpy.ndarray): Batch of segmentation predictions as tensor of floats.
        labels (numpy.ndarray): Batch of segmentation ground-truths as tensor of floats.

    Returns:
        tuple of int (TP, FP, FN, TN) across the entire batch.
    """
    pred_masks = pred_masks.astype(bool)
    labels = labels.astype(bool)
    tp = ((pred_masks == True) & (labels == True)).sum()
    fp = ((pred_masks == True) & (labels == False)).sum()
    fn = ((pred_masks == False) & (labels == True)).sum()
    tn = ((pred_masks == False) & (labels == False)).sum()


    return tp, fp, fn, tn

def accuracy(pred_masks,labels):
    """Compute pixel-level accuracy for binary semantic segmentation.

    Args:
        pred_masks (numpy.ndarray): Batch of segmentation predictions as tensor of floats.
        labels (numpy.ndarray): Batch of segmentation ground-truths as tensor of floats.

    Returns:
        acc (float): Pixel-level accuracy computed over the entire batch.
    """
    tp, fp, fn, tn = binary_all(pred_masks, labels)
    acc = (tp+tn)/(tp+tn+fp+fn)

    return acc

def precision(pred_masks,labels):
    """Compute pixel-level precision for binary semantic segmentation.

    Args:
        pred_masks (numpy.ndarray): Batch of segmentation predictions as tensor of floats.
        labels (numpy.ndarray): Batch of segmentation ground-truths as tensor of floats.

    Returns:
        pre (float): Pixel-level precision computed over the entire batch.
    """
    tp, fp, fn, tn = binary_all(pred_masks, labels)
    eps = 1e-7
    pre = tp/(tp+fp+eps)

    return pre

def recall(pred_masks,labels):
    """Compute pixel-level recall for binary semantic segmentation.

    Args:
        pred_masks (numpy.ndarray): Batch of segmentation predictions as tensor of floats.
        labels (numpy.ndarray): Batch of segmentation ground-truths as tensor of floats.

    Returns:
        rec (float): Pixel-level recall computed over the entire batch.
    """
    tp, fp, fn, tn = binary_all(pred_masks, labels)
    eps = 1e-7
    rec = tp/(tp+fn + eps)

    return rec

def main():
    pred_mask_dir = "/lustre/BIF/nobackup/hsieh005/train_patch/tb_UnetMit-b3_lr1e-4_Ldice_patch_ep100/tb_mask"
    label_dir = "/lustre/BIF/nobackup/hsieh005/train_patch/GT_label"
    output_csv = "metrics_patch_lr1e-4_p7_ep100.csv"
    patch = True

    # get all the mask files
    mask_files = sorted([f for f in os.listdir(pred_mask_dir) if f.endswith(".png")])
    results = []

    for filename in mask_files:
        pred_mask_path = os.path.join(pred_mask_dir, filename)
        label_path = os.path.join(label_dir, filename)

        # read predicted mask
        pred_mask = Image.open(pred_mask_path)
        pred_mask = (np.array(pred_mask) > 0).astype(np.uint8)

        if patch:
            # read label
            size = (1560, 1560)
            if os.path.exists(label_path):
                label = Image.open(label_path)
                label = (np.array(label) > 0).astype(np.uint8)
            else:
                label = np.zeros(size, dtype=np.int32)
        else:
            # read label
            size = (480, 480)
            if os.path.exists(label_path):
                label = Image.open(label_path)
                label = label.resize(size, resample=Image.NEAREST)
                label = (np.array(label) > 0).astype(np.uint8)
            else:
                label = np.zeros(size, dtype=np.int32)

        # calculate metrics
        acc = accuracy(pred_mask, label)
        iou = binary_jaccard(pred_mask, label)
        dice = binary_Dice(pred_mask, label)
        pre = precision(pred_mask, label)
        rec = recall(pred_mask, label)

        # turn results into csv
        results.append({
            "filename": filename,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "iou": iou,
            "dice":dice
        })

    # create csv file
    df = pd.DataFrame(results)
    mean_row = {
        "filename": "Average",
        "accuracy": df["accuracy"].mean(),
        "precision": df["precision"].mean(),
        "recall": df["recall"].mean(),
        "iou": df["iou"].mean(),
        "dice": df["dice"].mean()
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index = True)
    df.to_csv(output_csv, index=False)
    print(f"\n Evaluation results saved to '{output_csv}'")

if __name__ == "__main__":
    main()




