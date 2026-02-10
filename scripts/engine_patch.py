#!/usr/bin/env python3
"""
Functionality for training and testing for a patch-based model.

Adapted from Jacky To
"""

# Import statements
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.amp import autocast


# Define train step function per epoch
def train_step(model, dataloader, loss_fn, performance_fn, optimizer, scaler, device):
    """Trains a PyTorch model for a single epoch.

    Args:
        model (nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_fn (nn.Module): A PyTorch loss function to minimize.
        performance_fn (function): A function that calculates a performance metric.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        scaler (torch.cuda.amp.GradScaler): A PyTorch gradient scaler to help minimize gradient underflow.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training performance metrics.
        In the form (train_loss, train_performance). For example: (0.1112, 0.8743)

    """
    # Put model in train mode
    model.train()

    # Setup tdqm loop for progress bar over batches
    batch_loop = tqdm(dataloader, desc="Batches")

    # Always treating performance_fn as list to allow for multiple metrics
    if not isinstance(performance_fn, list):
        performance_fn = [performance_fn]

    # Setup train loss and train performance values
    train_loss = 0
    train_perform = [0] * len(performance_fn)

    # Loop through data loader data batches
    for batch, (data, labels) in enumerate(batch_loop):
        # Add channel dimension or class dimension to label if not present yet
        if (len(labels.shape) == 3) or (len(labels.shape) == 1):
            labels = labels.unsqueeze(1)

        # Send data to target device
        data, labels = data.float().to(device), labels.float().to(device)

        # 1. Forward pass
        with autocast(device_type=device.type):
            pred_logits = model(data)

            # 2. Calculate and accumulate loss
            loss = loss_fn(pred_logits, labels)
            train_loss += loss.item()

        # 3. Backward pass
        optimizer.zero_grad()  # Sets gradient to zero
        scaler.scale(loss).backward()  # Calculate gradient
        scaler.step(optimizer)  # Updates weights using gradient
        scaler.update()

        # Calculate and accumulate performance metrics across all batches
        for ind, fn in enumerate(performance_fn):
            # make sure the label and shape are matched
            batch_label = labels.reshape(pred_logits.shape)
            train_perform[ind] += fn(pred_logits, batch_label).item()

        # Update tqdm loop
        batch_loop.set_postfix(train_loss=f"{loss.item():.4f}")

    # Adjust metrics to get average loss and performance per batch
    train_loss = train_loss / len(dataloader)
    train_perform = [perform / len(dataloader) for perform in train_perform]

    return train_loss, train_perform


# Define test step function per epoch
def test_step(model, dataloader, loss_fn, performance_fn, device,
              patch_size = (480, 480), stride = (240, 240)):
    """Tests a PyTorch model with sliding window method.

    Args:
        model (nn.Module): A PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        loss_fn (nn.Module): A PyTorch loss function to calculate loss on the test data.
        performance_fn (function): A function that calculates a performance metric.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").
        patch_size (tuple): the size of window per patch
        stride (tuple): the distance of window moving per step

    Returns:
        A tuple of testing loss and testing performance metrics.
        In the form (test_loss, test_accuracy). For example: (0.0223, 0.8985)

    This function now expects full-resolution images from the dataloader
    (e.g., [1, 3, 1560, 1560]) and performs patch-based inference to
    calculate loss and performance on the entire reconstructed image.

    Assumes dataloader has batch_size=1.
    """
    # Put model in eval mode
    model.eval()

    # Always treating performance_fn as list to allow for multiple metrics
    if not isinstance(performance_fn, list):
        performance_fn = [performance_fn]

    # Setup test loss and test performance values
    test_loss = 0
    test_perform = [0] * len(performance_fn)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches (batch_size has to be 1)
        batch_loop = tqdm(dataloader, desc="Validation")
        for batch, (data, labels) in enumerate(batch_loop):
            # data shape is [1, 3, 1560, 1560]
            # labels shape is [1, 1, 1560, 1560]
            # Add channel dimension or class dimension to label if not present yet
            if (len(labels.shape) == 3) or (len(labels.shape) == 1):
                labels = labels.unsqueeze(1)

            # Send data to target device
            data, labels = data.float().to(device), labels.float().to(device)

            # --- Sliding Window ---
            B, C, H, W = data.shape
            patch_h, patch_w = patch_size
            stride_h, stride_w = stride

            # Infer output channels by running a dummy patch
            try:
                out_channels = model.out_channels
            except AttributeError:
                dummy_patch = torch.randn(1, C, patch_h, patch_w, device=device)
                out_channels = model(dummy_patch).shape[1]

            # build accumulate
            accum_logit_preds = torch.zeros((B, out_channels, H, W), device=device)
            normalization_map = torch.zeros((B, out_channels, H, W), device=device)

            # generate coords
            y_coords = sorted(list(set(list(range(0, H - patch_h, stride_h)) + [H - patch_h])))
            x_coords = sorted(list(set(list(range(0, W - patch_w, stride_w)) + [W - patch_w])))

            for y in y_coords:
                for x in x_coords:
                    y_end = y + patch_h
                    x_end = x + patch_w

                    data_patch = data[:, :, y:y_end, x:x_end]
                    patch_logit_preds = model(data_patch)

                    accum_logit_preds[:, :, y:y_end, x:x_end] += patch_logit_preds
                    normalization_map[:, :, y:y_end, x:x_end] += 1

            # get the full logits
            full_logit_preds = accum_logit_preds / (normalization_map + 1e-8)

            # 2. Calculate and accumulate loss
            loss = loss_fn(full_logit_preds, labels)
            test_loss += loss.item()

            # Calculate and accumulate performance
            for ind, fn in enumerate(performance_fn):
                batch_label = labels.reshape(full_logit_preds.shape)
                perform = fn(full_logit_preds, batch_label)
                test_perform[ind] += perform.item()

            batch_loop.set_postfix(val_loss=f"{loss.item():.4f}")

    # Adjust metrics to get average loss and performance per batch
    test_loss = test_loss / len(dataloader)
    test_perform = [perform / len(dataloader) for perform in test_perform]

    return test_loss, test_perform
