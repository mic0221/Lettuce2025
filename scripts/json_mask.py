#!/usr/bin/env python3
"""
Convert the labels from json format (from AnyLabeling) to mask.png
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def binary_poly2px(filepath, custom_size=None):
    """Convert polygon annotations from a LabelMe JSON file into a binary pixel mask.

    Args:
        filepath (str): Path to the json files
        custom_size (tuple of int):  Output size of the binary mask (H, W)

    Returns:
        numpy.ndarray: 2D binary mask where pixels inside annotated polygons are labeled as 1
        and background pixels as 0
    """
    poly_json = json.load(open(filepath, "r"))

    # Create empty mask
    if custom_size is None:
        mask = np.zeros((poly_json["imageHeight"], poly_json["imageWidth"]), dtype=np.uint8)
    else:
        mask = np.zeros(custom_size, dtype=np.uint8)

    # Draw polygons
    for shape in poly_json["shapes"]:
        poly_points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [poly_points], color=1)

    return mask

def convert_json_folder(json_dir, save_dir):
    """Save the files to the folder

    Args:
        json_dir (str): Path to the json files
        save_dir (str):  Path to the saved binary masks

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"[INFO] Found {len(json_files)} json files in {json_dir}")

    for jf in tqdm(json_files, desc="Converting"):
        mask = binary_poly2px(os.path.join(json_dir, jf))

        # Save as PNG with same basename
        save_path = os.path.join(save_dir, os.path.splitext(jf)[0] + ".png")
        cv2.imwrite(save_path, mask * 255)  # 乘 255 讓前景變白，背景黑
    print(f"[INFO] Masks saved in {save_dir}")

def main():
    json_dir = "/Users/xieminxi/Documents/WUR/thesis/server/label_images_3"
    save_dir = "/Users/xieminxi/Documents/WUR/thesis/server/fine-tuning/testset/third_round/GT_label"

    convert_json_folder(json_dir, save_dir)

if __name__ == "__main__":
    main()



