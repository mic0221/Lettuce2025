#!/usr/bin/env python3
"""
Convert the .fimg file into coloured images to assist GT labeling.
The thresholds of fimg values for different colors are adjustable.
"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def read_fimg(filename) -> np.ndarray:
    """Turns an Fimg value into a numpy nd.array

    Args:
        filename (str): name of the file that is to be opened

    Returns:
        numpy.ndarray: 2D array representing the fimg image
    """

    image = np.fromfile(filename, np.dtype("float32"))
    image = image[2:]
    image = np.reshape(image, newshape=(1024, 1360))

    return image

def process_for_overlay(img, out_path):
    """
    Process Fv/Fm image for overlay:
    - remove <0 values
    - blue for 0-0.5
    - red for 0.5–0.7
    - grayscale base (0–1)

    Args:
        img (np.ndarray): 2D array representing the fimg image
        out_path: Path to the saved images

    Returns:
        None
        Output are saved as PNG
    """
    img = np.where(img < 0, np.nan, img)

    # Get actual image size (height, width)
    h, w = img.shape

    # === Create figure that matches image pixel size exactly ===
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    # --- Base: grayscale 0–1 ---
    plt.imshow(img, cmap="gray", vmin=0.5, vmax=1, interpolation="nearest")

    # --- Overlay mask ---
    overlay = np.zeros((*img.shape, 4))
    # blue: <0.5
    overlay[img < 0.5] = [0, 0, 1, 0.3]
    # red: 0.5–0.7
    overlay[(img >= 0.5) & (img < 0.7)] = [1, 0, 0, 0.4]

    plt.imshow(overlay, interpolation="nearest")

    # --- Clean output ---
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)

    # === Save transparent PNG (no border, no resize) ===
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

def batch_process_fimg(input_dir, output_dir):
    """
    Process all .fimg files in input_dir and save processed PNGs to output_dir

    Args:
        input_dir (str): Path to .fimg files
        output_dir (str): Path to the saved images

    Returns:
        None
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.fimg"))
    print(f"[INFO] Found {len(files)} fimg files")

    for f in files:
        try:
            img = read_fimg(f)
            out_path = output_dir / (f.stem + ".png")
            process_for_overlay(img, out_path)
            print(f"[OK] Saved {out_path.name}")
        except Exception as e:
            print(f"[ERROR] Could not process {f.name}: {e}")

def main():
    INPUT_DIR = Path("/Users/xieminxi/Documents/WUR/thesis/server/label_FvFm_3")
    OUTPUT_DIR = Path("/Users/xieminxi/Documents/WUR/thesis/server/label_FvFm_3/img")
    batch_process_fimg(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()

