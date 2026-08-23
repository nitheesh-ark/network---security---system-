import torch
from torch.utils.data import DataLoader

from model.nidsnet import NIDSNet
from dataset import NIDSDataset
from trainer.trainer import Trainer


def main():

    ##########################################
    # Hyperparameters
    ##########################################

    BATCH_SIZE = 1024
    NUM_WORKERS = 8
    EPOCHS = 50

    ##########################################
    # Dataset
    ##########################################

    train_dataset = NIDSDataset("./data/processed/train.pt")
    val_dataset = NIDSDataset("./data/processed/val.pt")

    ##########################################
    # DataLoader
    ##########################################

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    ##########################################
    # Model
    ##########################################

    model = NIDSNet()

    ##########################################
    # Trainer
    ##########################################

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    ##########################################
    # Train
    ##########################################

    trainer.fit(epochs=EPOCHS)


if __name__ == "__main__":
    main()