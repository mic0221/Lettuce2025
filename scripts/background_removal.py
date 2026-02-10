#!/usr/bin/env python3
"""
Remove the backgrounds from raw images by dilated masks
The dilation size can be adjusted by the kernel size and number of iterations (default = (5,5)*5).
"""

# Import statements
import numpy as np
from PIL import Image
import os
import cv2

def remove_bg_with_binary_mask(image_folder, mask_folder, save_folder,
                               dilate_size = (5,5), dilate_iter=5):
    """
    Remove background from RGB images using binary segmentation masks.

    Args:
        image_folder (str): Path to original RGB images
        mask_folder (str): Path to binary plant masks with the same filenames as images
        save_folder (str): Path to the segmented plant images
        dilate_size (tuple): Kernel size used for mask dilation
        dilate_iter (int): Number of dilation iterations applied to the binary mask

    Returns:
        None
        Processed images are saved as PNG files.

    """
    os.makedirs(save_folder, exist_ok=True)

    for img in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img)
        mask_path = os.path.join(mask_folder, img)

        input_img = np.array(Image.open(img_path).convert("RGB"))
        binary_mask = np.array(Image.open(mask_path))

        # resize image
        target_size = (1560, 1560)
        # input_img = cv2.resize(input_img, target_size[::-1], interpolation=cv2.INTER_AREA)
        binary_mask = cv2.resize(binary_mask, target_size[::-1], interpolation=cv2.INTER_AREA)

        # 確保 mask 是 0/1
        mask = (binary_mask > 0).astype(np.uint8)

        # Add dilation
        if dilate_iter > 0:
            kernel = np.ones(dilate_size, np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

        plant_only = input_img * mask[..., None]  # 只保留前景
        save_name = os.path.splitext(img)[0] + ".png"
        save_path = os.path.join(save_folder, save_name)
        Image.fromarray(plant_only.astype(np.uint8)).save(save_path)

        print(f"[INFO] Saved -> {save_path}")



def main():
    image_dir = "/Users/xieminxi/Documents/WUR/thesis/server/fine-tuning/two_rounds/images_2rounds"
    mask_dir = "/Users/xieminxi/Documents/WUR/thesis/server/fine-tuning/two_rounds/plant_mask"
    save_dir = "/Users/xieminxi/Documents/WUR/thesis/server/train_patch/plant_seg_2r"

    remove_bg_with_binary_mask(image_dir, mask_dir, save_dir)

if __name__ == "__main__":
    main()