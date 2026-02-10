#!/usr/bin/env python3
"""
Preprocesses for cropping the individual plant from the raw image for use in model.
The argparse functionality requires 4 input:
    input: Path to rgb images
    output: Path to cropped images
    tray_file: Path to the tray file
    target_date: Date for the folder of rgb images

Adapted from Jacky To
"""

# Import statements
import os
import numpy as np
import pandas as pd
from skimage import io, color, util
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
import argparse


# Import supporting modules
import utils

# Remove alpha channel - adapted from Chris Dijkstra
def no_alpha(rgb_im):
    """Removes the alpha channel of an RGB image with 4 channels

    As color.rgba2rgb converts values to floats between 0 and 1

    Args:
        rgb_im (numpy.ndarray): 4 dimensional array representing an RGB image

    Returns:
        numpy.ndarray: 3 dimensional array representing an RGB image
    """
    assert rgb_im.shape[2] == 4, "Input RGB image doesn't have an alpha channel"

    # Blend alpha channel
    alphaless = color.rgba2rgb(rgb_im)

    # Convert values from float64 back to uint8
    alphaless = util.img_as_ubyte(alphaless)
    return alphaless


# Crop specific region - adapted from Chris Dijkstra
def crop_region(image, centre, shape):
    """Crops an image area of specified width and height around a central point

    Args:
        image (np.ndarray): matrix representing the image
        centre (tuple): contains the x and y coordinate of the centre as integers
        shape (tuple): contains the height and width of the subregion in pixels as integers

    Returns:
        The cropped region of the original image
    """
    shape_r = np.array(shape)
    shape_r[shape_r % 2 == 1] += 1
    if image.ndim == 2:
        crop = image[
            centre[1] - shape_r[1] // 2 : centre[1] + shape[1] // 2,
            centre[0] - shape_r[0] // 2 : centre[0] + shape[0] // 2,
        ]
    else:
        crop = image[
            centre[1] - shape_r[1] // 2 : centre[1] + shape[1] // 2,
            centre[0] - shape_r[0] // 2 : centre[0] + shape[0] // 2,
            :,
        ]
    return crop

def read_tray(filepath, date):
    """Read the tray file and convert it into a dictionary

    Args:
        filepath (str): Filepath to the tray file
        date (str): the target date

    Returns:
        a dictionary with column names as keys
    """
    df = pd.read_csv(filepath)
    df["Measuring Date"] = pd.to_datetime(df["Measuring Date"]).dt.strftime("%Y-%m-%d")
    df_oneday= df[df["Measuring Date"] == date]
    tray_dict = {col: df_oneday[col].to_numpy() for col in df_oneday.columns}
    return tray_dict

def count_tray(filepath, date):
    """Counts how many plants there are in a tray

    Args:
        filepath (str): Filepath to the tray file
        date (str): the specific date for cropping

    Returns:
        number of plants in a tray
    """
    df = pd.read_csv(filepath)
    df["Measuring Date"] = pd.to_datetime(df["Measuring Date"]).dt.strftime("%Y-%m-%d")
    df_oneday = df[df["Measuring Date"] == date]
    tray_ids = df_oneday["Tray ID"]
    count_dict = tray_ids.value_counts().to_dict()
    return count_dict

def convert_rgb_to_tray(filepath, rgb_name):
    """Retrieve the tray ID corresponding to a given RGB image.

    Args:
        filepath (str): Filepath to the tray file
        rgb_name (str): Filename of the RGB image.

    Returns:
         Tray ID associated with the given RGB image
    """
    df = pd.read_csv(filepath)
    row = df.loc[df["Image"] == rgb_name, "Tray ID"]
    if row.empty:
        raise KeyError(f"[ERROR] Image '{rgb_name}' not found in {filepath}")

    return row.iloc[0]


# Define crop functions
def indiv_crop(rgb_img, crop_size, dist_plants, num_plants):
    """Crop individual plants from the RGB images.

    Assumes that distance between plants is same along width and height.

    Args:
        rgb_img (np.ndarray): RGB image as np.ndarray.
        crop_size (tuple): Tuple of ints (width, height) with desired resolution of image crops.
        dist_plants (int): Distance in px between plants along width and height.
        num_plants (int): Number of plants in image, 4 or 5.

    Returns:
        list: Contains cropped RGB images as np.ndarrays.
    """
    # Calculate center coordinates of RGB
    center_x = int(rgb_img.shape[1] / 2 + 0.5)
    center_y = int(rgb_img.shape[0] / 2 + 0.5)

    # Crop for 4 plants
    if num_plants == 4:
        # Crop RGB
        rgb_area1 = crop_region(
            image=rgb_img,
            centre=(center_x, center_y - dist_plants),
            shape=crop_size,
        )
        rgb_area2 = crop_region(
            image=rgb_img,
            centre=(center_x - dist_plants, center_y),
            shape=crop_size,
        )
        rgb_area3 = crop_region(
            image=rgb_img,
            centre=(center_x + dist_plants, center_y),
            shape=crop_size,
        )
        rgb_area4 = crop_region(
            image=rgb_img,
            centre=(center_x, center_y + dist_plants),
            shape=crop_size,
        )

        # Compile crops into lists
        rgb_crops = [rgb_area1, rgb_area2, rgb_area3, rgb_area4]

    # Crop for 5 plants
    elif num_plants == 5:
        # Crop RGB
        rgb_area1 = crop_region(
            image=rgb_img,
            centre=(
                center_x - dist_plants,
                center_y - dist_plants,
            ),
            shape=crop_size,
        )
        rgb_area2 = crop_region(
            image=rgb_img,
            centre=(
                center_x + dist_plants,
                center_y - dist_plants,
            ),
            shape=crop_size,
        )
        rgb_area3 = crop_region(
            image=rgb_img,
            centre=(center_x, center_y),
            shape=crop_size,
        )
        rgb_area4 = crop_region(
            image=rgb_img,
            centre=(
                center_x - dist_plants,
                center_y + dist_plants,
            ),
            shape=crop_size,
        )
        rgb_area5 = crop_region(
            image=rgb_img,
            centre=(
                center_x + dist_plants,
                center_y + dist_plants,
            ),
            shape=crop_size,
        )

        # Compile crops into lists
        rgb_crops = [rgb_area1, rgb_area2, rgb_area3, rgb_area4, rgb_area5]

    # Return lists of cropped images
    return rgb_crops


# Define end-to-end overlay crop function for multi-processing
def path_crop(
    rgb_path,
    tray_path,
    date,
    tray_reg,
    crop_shape,
    crop_dist,
    rm_alpha=True,
    rgb_save_dir=None,
):
    """Crops individual plants from RGB image from filepath.

    This function combines everything from filepath to cropping so
    it can used for multi-processing or -threading.

    If one of the image files can't be read, the function returns None.

    Args:
        rgb_path (str): Filepath to RGB image.
        tray_path (str): Filepath to tray file
        date (str): the specific date for cropping
        tray_reg (dict): Dictionary of information from the tray registration file.
        crop_shape (tuple): Tuple of ints for shape of crop.
        crop_dist (int): Vertical/horizontal distance between plants on tray.
        rm_alpha (bool, optional): Removes alpha channel from RGB. Defaults to True.
        rgb_save_dir (str, optional): Directory to save RGB crops. Defaults to None.

    Returns:
        list: Contains cropped RGB images as np.ndarrays.
    """
    # Read file, if can't read one of the files, function returns nothing
    try:
        rgb = io.imread(rgb_path)
    except Exception as e:
        print(f"[ERROR] Could not read {rgb_path}: {e}")
        return []

    # Remove alpha from RGB image if desired and image has 4 channels
    if rm_alpha and rgb.shape[2] == 4:
        rgb = no_alpha(rgb)

    tray_count = count_tray(tray_path, date)

    # load the rgb_name and convert into the name with tray ID
    rgb_name = os.path.basename(rgb_path)
    tray_name = convert_rgb_to_tray(tray_path, rgb_name)

    num_plants = tray_count.get(tray_name)
    if num_plants not in (4, 5):
        print(f"[ERROR] Unsupported num_plants={num_plants} for tray {tray_name}")
        return []

    # for ind_trayID
    all_trayIDs = [str(x) for x in tray_reg["Tray ID"]]
    ind_trayID = [i for i, val in enumerate(all_trayIDs) if val == tray_name]

    # Crop RGB, Fm and FvFm crops in such a way that they overlap
    rgb_crops = indiv_crop(
        rgb, crop_size=crop_shape, dist_plants=crop_dist, num_plants=num_plants
    )

    # Save cropped images if desired, with plant names in filename
    if rgb_save_dir is not None:
        all_plantnames = tray_reg["Plant Name"]
        plantnames = all_plantnames[ind_trayID]

        # Count to add correct area number and plant name
        count = 0
        for rgb_crop in rgb_crops:
            new_name = f"{plantnames[count]}_20DAS.png"
            count += 1
            try:
                utils.save_img(rgb_crop, target_dir=rgb_save_dir, filename=new_name)
            except Exception as e:
                print(f"[ERROR] Save failed for {new_name}: {e}")

    return rgb_crops


def main():
    # Set config
    parser = argparse.ArgumentParser(description="Cropping the individual plant")
    parser.add_argument("input", help = "Path to rgb_dir")
    parser.add_argument("output", help = "Path to rgb_crop_dir")
    parser.add_argument("tray_file", help = "Path to the tray file")
    parser.add_argument("target_date", help="The target date")
    args = parser.parse_args()

    rgb_dir = args.input
    rgb_crop_dir = args.output
    CORES = 12
    CROP = True
    CROP_DIST = 736
    CROP_SHAPE = (1560, 1560)

    # Prepare to track skipped images
    skipped = []

    # Read tray registration for info on image files
    tray_reg = read_tray(args.tray_file, args.target_date)

    # Crop original images
    if CROP:
        # Only crop RGB images without downscaling for overlay
        # Retrieve and sort RGB filenames
        rgb_names = sorted(os.listdir(rgb_dir))

        # Create list of filepaths
        rgb_filepaths = []
        for rgb_name in rgb_names:
            # Join directory and filename to make filepath
            rgb_filepath = os.path.join(rgb_dir, rgb_name)

            # Append filepaths to lists of filepaths
            rgb_filepaths.append(rgb_filepath)

        # Cropping for overlay from filepaths of original images
        with Pool(processes=CORES) as pool:
            prepped_crop = partial(
                path_crop,
                tray_path=args.tray_file,
                date=args.target_date,
                tray_reg=tray_reg,
                crop_shape=CROP_SHAPE,
                crop_dist=CROP_DIST,
                rgb_save_dir=rgb_crop_dir,
            )
            process_iter = pool.imap(
                func=prepped_crop,
                iterable=rgb_filepaths,
            )
            process_loop = tqdm(
                process_iter,
                desc="RGB cropping",
                total=len(rgb_filepaths),
            )

            # Execute processes in order and collect skipped files
            # count = 0
            for crops in process_loop:  # processes are executed during loop
                # Track which files were skipped
                if crops is None:
                    skipped.append(rgb_names[0])

    # Save list of skipped images as text file in working directory
    skipped_path = os.path.join(os.getcwd(), "skipped_images.txt")
    with open(skipped_path, "w") as skip_file:
        for skip_name in skipped:
            skip_file.write(skip_name)
            skip_file.write("\n")


if __name__ == "__main__":
    main()
