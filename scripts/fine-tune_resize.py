#!/usr/bin/env python3
"""
fine-tune a pretrained model  with the resize-based approach from a previous project, check out:
    https://github.com/JackyTomato/LettuceTrain

The used config file in config_parser.py was called "config_resize.json".

Adapted from Jacky To
"""

# Import statements
import os
import gc
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import config_parser as cp

# Import supporting modules
import data_setup, engine, model_builder, utils

# Load model from saved model filename
def load_model(config_path, checkpoint, device="cuda", multi_gpu=False):
    """Loads a model from the filepath of the saved model.

    The model is loaded by parsing the corresponding saved config .json,
    using the config settings to initialize the model and subsequently
    loading the saved model state into the initialized model.

    Assumes that the saved config .json is in the same directory
    as the saved model. Additionally, assumes that the saved config
    is named "config_<model filename without file extension>.json".

    Args:
        config_path (str): File path to the .json file.
        checkpoint (str): File of saved model state including path if necessary.
        device (str, optional): Device to send model to, "cpu" or "cuda". Defaults to "cuda".
        multi_gpu (bool): If True, loads model on multiple GPUs, assuming device is "cuda".

    Returns:
        torch.nn.Module: Loaded model as a PyTorch nn.Module class.
    """

    # Parse config.json as dict
    config_dict = utils.parse_json(config_path)

    # Assign model settings from config
    MODEL_TYPE = eval(config_dict["MODEL_TYPE"])
    MODEL_NAME = config_dict["MODEL_NAME"]
    ENCODER_NAME = config_dict["ENCODER_NAME"]
    ENCODER_WEIGHTS = config_dict["ENCODER_WEIGHTS"]
    if ENCODER_WEIGHTS == "None":  # Allow for untrained encoder
        ENCODER_WEIGHTS = eval(ENCODER_WEIGHTS)
    N_CHANNELS = config_dict["N_CHANNELS"]
    N_CLASSES = config_dict["N_CLASSES"]
    DECODER_ATTENTION = config_dict["DECODER_ATTENTION"]
    if DECODER_ATTENTION == "None":  # Allow for no decoder attention
        DECODER_ATTENTION = eval(DECODER_ATTENTION)
    ENCODER_FREEZE = eval(config_dict["ENCODER_FREEZE"])
    print(f"[INFO] Loading config.json was succesful!")

    # Initialize model and send to device
    model = MODEL_TYPE(
        model_name=MODEL_NAME,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        decoder_attention=DECODER_ATTENTION,
        encoder_freeze=ENCODER_FREEZE,
    )
    if multi_gpu:
        model = nn.DataParallel(model)
    if "cuda:0" in device:
        torch.cuda.set_device(device)
        model = model.cuda()

    print("[INFO] Model initialized!")

    # Load saved model state into freshly initialized model
    utils.load_checkpoint(checkpoint, model)
    return model

def main():
    torch.set_num_threads(1)

    # Reproducibility
    torch.manual_seed(cp.SEED)
    random.seed(cp.SEED)
    np.random.seed(cp.SEED)

    # ======================
    # CUSTOMIZE FOR FINETUNE
    # ======================
    cp.IMG_DIR = "/lustre/BIF/nobackup/hsieh005/fine-tuning/images"
    cp.LABEL_DIR = "/lustre/BIF/nobackup/hsieh005/fine-tuning/tipburn_relabel"
    cp.BATCH_SIZE = 4
    cp.LEARNING_RATE = 1e-5
    cp.LOAD_MODEL = True
    cp.LOAD_MODEL_PATH = "/lustre/BIF/nobackup/hsieh005/fine-tuning/tb_UnetMit-b3_lr1e-4_b32_Ldice_ep100.pth.tar"  # pretrained model (weight)
    cp.SAVE_MODEL_DIR = "/lustre/BIF/nobackup/hsieh005/fine-tuning/new_model_bg_cv"
    cp.SAVE_MODEL_NAME = "finetuned_model_bg.pth.tar"
    cp.ENCODER_FREEZE = False
    cp.NUM_EPOCHS = 100

    # Normal training, no K-fold cross validation
    if cp.KFOLD is None:
        # Load data
        loaders = data_setup.get_loaders(
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
        train_loader, test_loader = loaders
        print("[INFO] Fine-tuning dataset loaded!")

        # Load model
        model_config = "/lustre/BIF/nobackup/hsieh005/fine-tuning/config_tb_UnetMit-b3_lr1e-4_b32_Ldice_ep100.json"
        model = load_model(model_config, cp.LOAD_MODEL_PATH, device=cp.DEVICE, multi_gpu=cp.MULTI_GPU)

        # Optionally freeze encoder
        if cp.ENCODER_FREEZE:
            for param in model.module.model.encoder.parameters():
                param.requires_grad = False
            print("[INFO] Encoder frozen for fine-tuning")

        # Optimizer
        optimizer = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

        # Training loop
        results = {"epoch": [], "train_loss": [], "train_perform": [],
                   "test_loss": [], "test_perform": []}

        # Setup tqdm loop for progress bar over epochs
        epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Fine-tuning Epochs")
        for epoch in epoch_loop:
            train_loss, train_perform = engine.train_step(
                model=model,
                dataloader=train_loader,
                loss_fn=cp.LOSS_FN,
                performance_fn=cp.PERFORMANCE_FN,
                optimizer=optimizer,
                scaler=cp.SCALER,
                device=cp.DEVICE,
            )
            test_loss, test_perform = engine.test_step(
                model=model,
                dataloader=test_loader,
                loss_fn=cp.LOSS_FN,
                performance_fn=cp.PERFORMANCE_FN,
                device=cp.DEVICE,
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

        # Save final fine-tuned model
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
            model=model,
            target_dir=cp.SAVE_MODEL_DIR,
            filename=f"summary_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.txt",
            n_channels=cp.N_CHANNELS,
        )

        # Save the config
        utils.save_config(
            target_dir=cp.SAVE_MODEL_DIR,
            filename=f"config_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.json",
        )

        print("[INFO] Fine-tuning complete!")

    else:
        print(f"[INFO] Performing {cp.KFOLD}-fold cross-validation")

        # Create DataLoaders with help from data_setup.py
        loaders = data_setup.get_loaders(
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
        print("[INFO] Fine-tuning dataset loaded!")

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

            # Load model
            model_config = "/lustre/BIF/nobackup/hsieh005/fine-tuning/config_tb_UnetMit-b3_lr1e-4_b32_Ldice_ep100.json"
            model = load_model(model_config, cp.LOAD_MODEL_PATH, device=cp.DEVICE, multi_gpu=cp.MULTI_GPU)

            # Optionally freeze encoder
            if cp.ENCODER_FREEZE:
                for param in model.module.model.encoder.parameters():
                    param.requires_grad = False
                print("[INFO] Encoder frozen for fine-tuning")

            # Optimizer
            optimizer = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

            # Training loop
            results = {"epoch": [], "train_loss": [], "train_perform": [],
                       "test_loss": [], "test_perform": []}

            # Create model save name
            model_name_split = cp.SAVE_MODEL_NAME.split(os.extsep, 1)
            model_new_name = (
                f"{model_name_split[0]}_fold{fold + 1}.{model_name_split[1]}"
            )

            # Setup tqdm loop for progress bar over epochs
            epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Fine-tuning Epochs")
            for epoch in epoch_loop:
                train_loss, train_perform = engine.train_step(
                    model=model,
                    dataloader=train_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    optimizer=optimizer,
                    scaler=cp.SCALER,
                    device=cp.DEVICE,
                )
                test_loss, test_perform = engine.test_step(
                    model=model,
                    dataloader=test_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    device=cp.DEVICE,
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
                            model_name=model_new_name,
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

            # Save final fine-tuned model
            if cp.CHECKPOINT_FREQ is None:
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=model_new_name,
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
                    model_name=model_new_name,
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
                n_channels=cp.N_CHANNELS,
            )

            # Save the config
            utils.save_config(
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"config_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.json",
            )

            print(f"[INFO] Fine-tuning with Fold {fold + 1} finished!")


if __name__ == "__main__":
    main()
