#!/usr/bin/env python3
"""
Functionality for creating lettuce dataset as PyTorch Dataset
and loading the train & test datasets for a resize-based model.

Adapted from Jacky To
"""
# Import statements
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split, KFold

# Import supporting modules
import utils


# Define Pytorch Dataset class for lettuce segmentation dataset
class LettuceSegDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        is_train,
        train_frac=0.75,
        kfold=None,
        transform=None,
        seed=42,
        give_name=False,
    ):
        """Creates a PyTorch Dataset class of the lettuce segmantation dataset.

        Uses sorted() to match filenames of images and masks. Check if filenames
        are constructed in such a way that sorted() will sort the filenames in the same way.
        Otherwise images and masks will be mismatched when loading the data.

        Allows for a training fraction of 1, in that case the whole dataset is provided.
        When doing K-fold cross valiation, the whole dataset is also provided,
        as the split will be performed downstream.

        Args:
            img_dir (str): Filepath of directory containing the images.
            label_dir (str): Filepath of directory  containing the segmentation masks.
            is_train (bool): If true, gives train data. If false, gives test data.
            train_frac (float, optional): Fraction of data that is train. Defaults to 0.75.
            kfold (int, optional): K for K-fold cross validation. If not None, train_frac is ignored. Defaults to None.
            transform (albumentations.Compose, optional): Transformations for data aug. Defaults to None.
            seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.
            give_name (bool, optional): If True, dataset also provides image name.
        """
        self.transform = transform
        self.give_name = give_name

        # List all image and mask filenames
        self.img_names = sorted(os.listdir(img_dir))
        mask_names = sorted(os.listdir(label_dir))


        # Check if there is an incomplete number of masks
        if len(self.img_names) != len(mask_names):
            incomplete_masks = True
            print(
                "[INFO] Numbers of images and masks are inequal, cancel if unintended!"
            )
        else:
            incomplete_masks = False

        # Create lists of filepaths for images and masks
        if incomplete_masks is False:
            img_paths = []
            mask_paths = []
            for img_name, mask_name in zip(self.img_names, mask_names):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(label_dir, mask_name)
                img_paths.append(img_path)
                mask_paths.append(mask_path)

        # Create lists of filepaths for images and masks when there are not masks for every image
        else:
            img_paths = []
            mask_paths = []
            mask_names = np.array(mask_names)
            for img_name in self.img_names:
                img_path = os.path.join(img_dir, img_name)
                img_paths.append(img_path)

                # List raw image name in mask paths for missing masks
                raw_name = img_name.split(os.extsep)[0]
                if raw_name.endswith("_bg_mask"):  # for filenames of bg masked images
                    raw_name = raw_name.removesuffix("_bg_mask")

                match_ind = np.flatnonzero(
                    np.core.defchararray.find(mask_names, raw_name) != -1
                )
                if len(match_ind) == 1:
                    mask_path = os.path.join(label_dir, mask_names[match_ind[0]])
                elif len(match_ind) == 0:
                    mask_path = raw_name
                mask_paths.append(mask_path)


        # Split into train and test sets if desired when not doing K-fold cross validation
        if (train_frac < 1) and (kfold is None):
            split = train_test_split(
                img_paths,
                mask_paths,
                train_size=train_frac,
                random_state=seed,
            )

            # Give train or test data as requested
            if is_train:
                self.img_paths = split[0]
                self.mask_paths = split[2]
            else:
                self.img_paths = split[1]
                self.mask_paths = split[3]

        # Don't split when training fraction is 1 or when doing K-fold cross validation
        else:
            self.img_paths = img_paths
            self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Retrieve image, should be np.array for albumentations.transforms
        img = np.array(Image.open(self.img_paths[index]))

        # Retrieve mask, mask could be .json or an image format
        size = img.shape[:2]
        if self.mask_paths[index].endswith(".json"):
            mask = utils.binary_poly2px(self.mask_paths[index], custom_size=size)

        elif self.mask_paths[index].endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ):
            mask = np.array(Image.open(self.mask_paths[index]))
            mask = (mask > 0).astype(np.int32)
        else:
            mask = np.zeros(size, dtype=np.int32)  # Create empty mask for missing mask
        if 255.0 in mask:
            mask[mask == 255.0] = 1.0

        # Apply data augmentation transforms to image and mask
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)

            # Retrieve augmentations
            mask = augmentations["mask"]
            img = augmentations["image"]

        result = [img, mask]

        # Also provide image name if desired
        if self.give_name:
            img_name = os.path.basename(os.path.normpath(self.img_paths[index]))
            result.append(img_name)

        return result

# Define data loaders for training and testing
def get_loaders(
    dataset,
    img_dir,
    label_dir,
    train_augs,
    test_augs,
    batch_size,
    num_workers,
    train_frac=0.75,
    kfold=None,
    pin_memory=True,
    seed=42,
):
    """Creates PyTorch DataLoaders for train and test dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset class inherited from PyTorch's Dataset class.
        img_dir (string): Path of directory containing the image data.
        label_dir (string): Path of directory containing the mask_paths of the image data.
        train_augs (albumentations.Compose/transforms.Compose): Albumentations or PyTorch transforms for train.
        test_augs (albumentations.Compose/transforms.Compose): Albumentations or PyTorch transforms for test.
        batch_size (int): Number of samples in each batch.
        num_workers (int): Number of worker processes for data loading.
        train_frac (float, optional): Fraction of data to be used for training. Defaults to 0.75.
        kfold (int, optional): K for K-fold cross validation. If not None, train_frac is ignored. Defaults to None.
        pin_memory (bool, optional): Speeds up data transfer from CPU to GPU. Defaults to True.
        seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.

    Returns:
        tuple: Contains torch.utils.data.DataLoader objects for training and testing dataset.
        Or when doing K-fold cross validation:
        list: Containing tuples of torch.utils.data.DataLoader objects for training and testing dataset.
    """
    if kfold is None:
        # Get train and test datasets
        train_ds = dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            train_frac=train_frac,
            is_train=True,
            transform=train_augs,
            seed=seed,
        )
        test_ds = dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            train_frac=train_frac,
            is_train=False,
            transform=test_augs,
            seed=seed,
        )

        # Create DataLoaders of datasets
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
        loaders = (train_loader, test_loader)

    # Create dataloaders for each split of K-fold cross validation
    else:
        # Get full to be split into K-folds dataset
        full_ds = dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            train_frac=train_frac,
            kfold=kfold,
            is_train=True,
            transform=train_augs,
            seed=seed,
        )

        # Create generator of train and set indices for each fold
        kfolder = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        inds_kfold = kfolder.split(full_ds)

        # Create list of train and test DataLoader objects
        loaders = []
        for train_inds, test_inds in inds_kfold:
            # Create DataLoaders for current fold and add to list
            train_loader = DataLoader(
                full_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_inds,
            )
            test_loader = DataLoader(
                full_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=test_inds,
            )
            loaders.append((train_loader, test_loader))

    return loaders


# MNIST handwritten digit dataset for testing classification
def MNIST_digit_loaders(batch_size, num_workers, pin_memory=True):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/lustre/BIF/nobackup/to001/thesis_MBF/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/lustre/BIF/nobackup/to001/thesis_MBF/data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader
