#!/usr/bin/env python3
"""
fine-tune a pretrained model with the patch-based approach from a previous project, check out:
    https://github.com/JackyTomato/LettuceTrain

The used config file in config_parser.py was called "config_patch.json".

Adapted from Jacky To
"""

# Import statements
import os
import torch
import torch.nn as nn
import numpy as np
import random
import gc
from tqdm import tqdm

print("[INFO] Libraries loaded!")

import data_setup_patch, engine_patch, model_builder, utils
import config_parser as cp

print("[INFO] Loading config_patch.json was successful!")


def main():
    # Set cap on thread usage
    torch.set_num_threads(1)

    # Set seeds for reproducibility
    torch.manual_seed(cp.SEED)
    random.seed(cp.SEED)
    np.random.seed(cp.SEED)

    # Normal training, no K-fold cross validation
    if cp.KFOLD is None:
            # Create DataLoaders with help from data_setup_patch.py
            loaders = data_setup_patch.get_loaders(
                dataset=cp.DATASET,
                img_dir=cp.IMG_DIR,
                label_dir=cp.LABEL_DIR,
                train_frac=cp.TRAIN_FRAC,
                kfold=cp.KFOLD,
                train_augs=cp.TRAIN_TRANSFORMS,
                test_augs=cp.TEST_TRANSFORMS,
                batch_size=cp.BATCH_SIZE,
                num_workers=cp.NUM_WORKERS,
                pin_memory=cp.PIN_MEMORY,
                seed=cp.SEED,
            )
            print("[INFO] Data successfully loaded!")

            # (Re-)initialize model for next fold with help from model_builder.py and send to device
            model = cp.MODEL_TYPE(
                model_name=cp.MODEL_NAME,
                encoder_name=cp.ENCODER_NAME,
                encoder_weights=cp.ENCODER_WEIGHTS,
                n_channels=cp.N_CHANNELS,
                n_classes=cp.N_CLASSES,
                decoder_attention=cp.DECODER_ATTENTION,
                encoder_freeze=cp.ENCODER_FREEZE
            )
            if cp.MULTI_GPU:
                model = nn.DataParallel(model)
            model = model.to(cp.DEVICE)

            # Load model if requested
            if cp.LOAD_MODEL:
                utils.load_checkpoint(checkpoint=cp.LOAD_MODEL_PATH, model=model)

            # Prepare optimizer
            optimizer = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

            # Create empty results dictionary for loss and performance during training loop
            results = {
                "epoch": [],
                "train_loss": [],
                "train_perform": [],
                "test_loss": [],
                "test_perform": [],
            }

            # Prepare loaders for loop over epochs without K-fold CV
            train_loader, test_loader = loaders

            # Setup tqdm loop for progress bar over epochs
            epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Epochs")
            for epoch in epoch_loop:
                train_loss, train_perform = engine_patch.train_step(
                    model=model,
                    dataloader=train_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    optimizer=optimizer,
                    scaler=cp.SCALER,
                    device=cp.DEVICE,
                )
                
                test_loss, test_perform = engine_patch.test_step(
                    model=model.module,
                    dataloader=test_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    device=cp.DEVICE,
                    patch_size=(480, 480),
                    stride=(240, 240)
                )

                # Checkpoint model at a given frequency if requested
                if cp.CHECKPOINT_FREQ is not None:
                    if (
                        epoch + 1
                    ) % cp.CHECKPOINT_FREQ == 0:  # Current epoch is epoch + 1
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        utils.save_checkpoint(
                            state=checkpoint,
                            target_dir=cp.SAVE_MODEL_DIR,
                            model_name=cp.SAVE_MODEL_NAME,
                        )

                # Print out epoch number, loss and performance for this epoch
                print(
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_perform: {', '.join(f'{perform:.4f}' for perform in train_perform)} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_perform: {', '.join(f'{perform:.4f}' for perform in test_perform)}"
                )

                # Update results dictionary
                results["epoch"].append(epoch + 1)
                results["train_loss"].append(train_loss)
                results["train_perform"].append(train_perform)
                results["test_loss"].append(test_loss)
                results["test_perform"].append(test_perform)

            # Save the model with help from utils.py
            if cp.CHECKPOINT_FREQ is None:
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=cp.SAVE_MODEL_NAME,
                )
            elif (
                cp.NUM_EPOCHS % cp.CHECKPOINT_FREQ != 0
            ):  # Don't save when final epoch was checkpoint
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=cp.SAVE_MODEL_NAME,
                )

            # Save loss and performance during training
            utils.save_train_results(
                dict_results=results,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"results_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.tsv",
            )


            utils.save_network_summary(
                model=model.module,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"summary_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.txt",
                n_channels=cp.N_CHANNELS
            )

            # Save the config
            utils.save_config(
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"config_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.json",
                config_name="config_patch.json"
            )


    # Perform K-fold cross validation
    else:
        print(f"[INFO] Performing {cp.KFOLD}-fold cross-validation")

        # Create DataLoaders with help from data_setup_patch.py
        loaders = data_setup_patch.get_loaders(
            dataset=cp.DATASET,
            img_dir=cp.IMG_DIR,
            label_dir=cp.LABEL_DIR,
            train_frac=cp.TRAIN_FRAC,
            kfold=cp.KFOLD,
            train_augs=cp.TRAIN_TRANSFORMS,
            test_augs=cp.TEST_TRANSFORMS,
            batch_size=cp.BATCH_SIZE,
            num_workers=cp.NUM_WORKERS,
            pin_memory=cp.PIN_MEMORY,
            seed=cp.SEED,
        )
        print("[INFO] Data succesfully loaded!")

        # Setup tqdm loop for progress bar over K-folds
        kfold_loop = tqdm(loaders, desc="Cross Validation Folds")
        for fold, (train_loader, test_loader) in enumerate(kfold_loop):
            # Clean up old objects and free up GPU memory if not first fold
            if fold > 0:
                del model, optimizer, results
                gc.collect()
                if cp.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"[INFO] Re-initializing model, optimizer and results logger for fold {fold + 1}!"
                )

            # (Re-)initialize model for next fold with help from model_builder.py and send to device
            model = cp.MODEL_TYPE(
                model_name=cp.MODEL_NAME,
                encoder_name=cp.ENCODER_NAME,
                encoder_weights=cp.ENCODER_WEIGHTS,
                n_channels=cp.N_CHANNELS,
                n_classes=cp.N_CLASSES,
                decoder_attention=cp.DECODER_ATTENTION,
                encoder_freeze=cp.ENCODER_FREEZE
            )
            if cp.MULTI_GPU:
                model = nn.DataParallel(model)
            model = model.to(cp.DEVICE)

            # Load model if requested
            if cp.LOAD_MODEL:
                utils.load_checkpoint(checkpoint=cp.LOAD_MODEL_PATH, model=model)

            # Prepare optimizer
            optimizer = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

            # Create empty results dictionary for loss and performance during training loop
            results = {
                "epoch": [],
                "train_loss": [],
                "train_perform": [],
                "test_loss": [],
                "test_perform": [],
            }

            # Create model save name
            model_name_split = cp.SAVE_MODEL_NAME.split(os.extsep, 1)
            model_new_name = (
                f"{model_name_split[0]}_fold{fold + 1}.{model_name_split[1]}"
            )

            # Setup tqdm loop for progress bar over epochs
            epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Epochs")
            for epoch in epoch_loop:
                train_loss, train_perform = engine_patch.train_step(
                    model=model,
                    dataloader=train_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    optimizer=optimizer,
                    scaler=cp.SCALER,
                    device=cp.DEVICE,
                )
                test_loss, test_perform = engine_patch.test_step(
                    model=model,
                    dataloader=test_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    device=cp.DEVICE,
                    patch_size=(480, 480),
                    stride=(240, 240)
                )

                # Checkpoint model at a given frequency if requested
                if cp.CHECKPOINT_FREQ is not None:
                    if (
                        epoch + 1
                    ) % cp.CHECKPOINT_FREQ == 0:  # Current epoch is epoch + 1
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        utils.save_checkpoint(
                            state=checkpoint,
                            target_dir=cp.SAVE_MODEL_DIR,
                            model_name=cp.SAVE_MODEL_NAME,
                        )

                # Print out epoch number, loss and performance for this epoch
                print(
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_perform: {', '.join(f'{perform:.4f}' for perform in train_perform)} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_perform: {', '.join(f'{perform:.4f}' for perform in test_perform)}"
                )

                # Update results dictionary
                results["epoch"].append(epoch + 1)
                results["train_loss"].append(train_loss)
                results["train_perform"].append(train_perform)
                results["test_loss"].append(test_loss)
                results["test_perform"].append(test_perform)

            # Save the model with help from utils.py
            if cp.CHECKPOINT_FREQ is None:
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=cp.SAVE_MODEL_NAME,
                )
            elif (
                cp.NUM_EPOCHS % cp.CHECKPOINT_FREQ != 0
            ):  # Don't save when final epoch was checkpoint
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=cp.SAVE_MODEL_NAME,
                )

            # Save loss and performance during training
            utils.save_train_results(
                dict_results=results,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"results_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.tsv",
            )

            
            utils.save_network_summary(
                model=model,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"summary_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.txt",
                n_channels=cp.N_CHANNELS
            )

            # Save the config
            utils.save_config(
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"config_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.json",
                config_name="config_patch.json"
            )

            print(f"[INFO] Fold {fold + 1} finished!")


if __name__ == "__main__":
    main()
