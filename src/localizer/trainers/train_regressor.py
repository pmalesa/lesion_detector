import os
import logging

import torch
import torch.nn as nn
from common.file_utils import load_metadata
from torch.utils.data import DataLoader
from localizer.networks.regression import BoxRegressor
from pathlib import Path
from common.dataset_loader import LesionDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("LESION-DETECTOR")
def train_regressor(config):
    logger.info("Starting backbone CNN fine-tuning on regression task.")

    # Load metadata and parameters
    dataset_metadata = load_metadata(config.get("metadata_path", ""))
    images_dir = config.get("images_dir", "")
    models_dir = Path(config.get("models_dir", "models/"))
    os.makedirs(models_dir, exist_ok=True)
    backbone_cnn_path = Path(config.get("backbone_cnn_path", ""))
    config = config["regression"]
    epochs = config.get("epochs", 10)
    log_interval = config.get("log_interval", 1)

    # Prepare dataset and loaders
    train_dataset = LesionDataset(split="train", metadata=dataset_metadata, images_dir=images_dir)
    val_dataset = LesionDataset(split="validation", metadata=dataset_metadata, images_dir=images_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Prepare model, optimizer and loss criterion
    model = BoxRegressor().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # Optimize only unfrozen parameters
        lr = config["learning_rate"]
    )
    criterion = nn.SmoothL1Loss()

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                logger.info(
                    f"Epoch: {epoch + 1}/{epochs} - Batch: {i + 1}/{len(train_loader)} "
                    f"- Loss: {loss:.4f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                logger.info(f"Validation in progress - Batch: {i + 1}/{len(val_loader)}")

        val_loss /= len(val_loader)

        logger.info(f"Epoch: {epoch + 1} - Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_backbone(str(backbone_cnn_path))
            logger.info(f"Best backbone CNN saved to '{str(backbone_cnn_path)}'.")
    
    logger.info("Finished backbone CNN fine-tuning on regression task.")
