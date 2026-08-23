"""
trainer.py

Professional Trainer for NIDSNet

Features
--------
✓ Mixed Precision (AMP)
✓ Gradient Clipping
✓ Validation
✓ Checkpoint Saving
✓ Best Model Saving
✓ Accuracy
✓ Progress Bars
✓ Learning Rate Scheduler
✓ Resume Training
"""

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer=None,
        scheduler=None,
        criterion=None,
        device=None,
        checkpoint_dir="checkpoints",
        max_grad_norm=1.0,
        use_amp=True
    ):

        ##################################################
        # Device
        ##################################################

        if device is None:

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        ##################################################
        # Model
        ##################################################

        self.model = model.to(device)

        ##################################################
        # Data
        ##################################################

        self.train_loader = train_loader
        self.val_loader = val_loader

        ##################################################
        # Loss
        ##################################################

        if criterion is None:

            criterion = nn.CrossEntropyLoss()

        self.criterion = criterion

        ##################################################
        # Optimizer
        ##################################################

        if optimizer is None:

            optimizer = AdamW(

                self.model.parameters(),

                lr=1e-3,

                weight_decay=1e-4

            )

        self.optimizer = optimizer

        ##################################################
        # Scheduler
        ##################################################

        if scheduler is None:

            scheduler = CosineAnnealingLR(

                self.optimizer,

                T_max=50

            )

        self.scheduler = scheduler

        ##################################################
        # AMP
        ##################################################

        self.use_amp = (
            use_amp and
            self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        ##################################################
        # Misc
        ##################################################

        self.max_grad_norm = max_grad_norm

        self.start_epoch = 0

        self.best_accuracy = 0.0

        ##################################################
        # Checkpoints
        ##################################################

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    ##################################################
# Train One Epoch
##################################################

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(

            self.train_loader,

            desc="Training",

            leave=False

        )

        for batch in progress:

            ##################################################
            # Move batch to GPU
            ##################################################

            flow = batch["flow"].to(
                self.device,
                non_blocking=True
            )

            packet = batch["packet"].to(
                self.device,
                non_blocking=True
            )

            time = batch["time"].to(
                self.device,
                non_blocking=True
            )

            flags = batch["flags"].to(
                self.device,
                non_blocking=True
            )

            header = batch["header"].to(
                self.device,
                non_blocking=True
            )

            labels = batch["label"].to(
                self.device,
                non_blocking=True
            )

            ##################################################
            # Zero Gradients
            ##################################################

            self.optimizer.zero_grad(
                set_to_none=True
            )

            ##################################################
            # Forward
            ##################################################

            with torch.autocast(

                device_type=self.device.type,

                enabled=self.use_amp

            ):

                logits = self.model(

                    flow,

                    packet,

                    time,

                    flags,

                    header

                )

                loss = self.criterion(

                    logits,

                    labels

                )

            ##################################################
            # Backward
            ##################################################

            self.scaler.scale(

                loss

            ).backward()

            ##################################################
            # Gradient Clipping
            ##################################################

            self.scaler.unscale_(

                self.optimizer

            )

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.max_grad_norm

            )

            ##################################################
            # Optimizer
            ##################################################

            self.scaler.step(

                self.optimizer

            )

            self.scaler.update()

            ##################################################
            # Statistics
            ##################################################

            running_loss += loss.item()

            predictions = torch.argmax(

                logits,

                dim=1

            )

            correct += (

                predictions == labels

            ).sum().item()

            total += labels.size(0)

            ##################################################
            # Progress Bar
            ##################################################

            progress.set_postfix(

                loss=f"{loss.item():.4f}",

                acc=f"{100*correct/total:.2f}%"

            )

        epoch_loss = running_loss / len(

            self.train_loader

        )

        epoch_accuracy = (

            100.0 * correct / total

        )

        return epoch_loss, epoch_accuracy

    ##################################################
# Validation
##################################################

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            progress = tqdm(

                self.val_loader,

                desc="Validation",

                leave=False

            )

            for batch in progress:

                ##################################################
                # Move Batch to GPU
                ##################################################

                flow = batch["flow"].to(
                    self.device,
                    non_blocking=True
                )

                packet = batch["packet"].to(
                    self.device,
                    non_blocking=True
                )

                time = batch["time"].to(
                    self.device,
                    non_blocking=True
                )

                flags = batch["flags"].to(
                    self.device,
                    non_blocking=True
                )

                header = batch["header"].to(
                    self.device,
                    non_blocking=True
                )

                labels = batch["label"].to(
                    self.device,
                    non_blocking=True
                )

                ##################################################
                # Forward Only
                ##################################################

                with torch.autocast(

                    device_type=self.device.type,

                    enabled=self.use_amp

                ):

                    logits = self.model(

                        flow,

                        packet,

                        time,

                        flags,

                        header

                    )

                    loss = self.criterion(

                        logits,

                        labels

                    )

                ##################################################
                # Statistics
                ##################################################

                running_loss += loss.item()

                predictions = torch.argmax(

                    logits,

                    dim=1

                )

                correct += (

                    predictions == labels

                ).sum().item()

                total += labels.size(0)

                ##################################################
                # Progress Bar
                ##################################################

                progress.set_postfix(

                    loss=f"{loss.item():.4f}",

                    acc=f"{100*correct/total:.2f}%"

                )

        epoch_loss = running_loss / len(

            self.val_loader

        )

        epoch_accuracy = (

            100.0 * correct / total

        )

        return epoch_loss, epoch_accuracy





    ##################################################
# Save Checkpoint
##################################################

    def save_checkpoint(
        self,
        epoch,
        is_best=False
    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_accuracy":
                self.best_accuracy

        }

        latest_path = (
            self.checkpoint_dir /
            "latest.pt"
        )

        torch.save(
            checkpoint,
            latest_path
        )

        if is_best:

            best_path = (
                self.checkpoint_dir /
                "best_model.pt"
            )

            torch.save(
                checkpoint,
                best_path
            )

            print(
                "\n✓ Best model saved."
            )
    ##################################################
# Load Checkpoint
##################################################

    def load_checkpoint(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]

        )

        self.optimizer.load_state_dict(

            checkpoint[
                "optimizer_state_dict"
            ]

        )

        self.scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]

        )

        self.best_accuracy = checkpoint[
            "best_accuracy"
        ]

        self.start_epoch = (
            checkpoint["epoch"] + 1
        )

        print(
            f"Checkpoint loaded "
            f"(Epoch {self.start_epoch})"
        )







            ##################################################
    # Main Training Loop
    ##################################################

    def fit(
        self,
        epochs
    ):

        print("\nTraining Started\n")

        for epoch in range(

            self.start_epoch,

            epochs

        ):

            print("=" * 70)

            print(

                f"Epoch "

                f"{epoch+1}/{epochs}"

            )

            print("=" * 70)

            ##########################################

            train_loss, train_acc = (

                self.train_one_epoch()

            )

            ##########################################

            val_loss, val_acc = (

                self.validate()

            )

            ##########################################

            self.scheduler.step()

            ##########################################

            if val_acc > self.best_accuracy:

                self.best_accuracy = val_acc

                self.save_checkpoint(

                    epoch,

                    is_best=True

                )

            else:

                self.save_checkpoint(

                    epoch,

                    is_best=False

                )

            ##########################################

            lr = (

                self.optimizer

                .param_groups[0]["lr"]

            )

            ##########################################

            print()

            print(

                f"Train Loss : "

                f"{train_loss:.4f}"

            )

            print(

                f"Train Acc  : "

                f"{train_acc:.2f}%"

            )

            print()

            print(

                f"Val Loss   : "

                f"{val_loss:.4f}"

            )

            print(

                f"Val Acc    : "

                f"{val_acc:.2f}%"

            )

            print()

            print(

                f"Best Acc   : "

                f"{self.best_accuracy:.2f}%"

            )

            print(

                f"LR         : "

                f"{lr:.7f}"

            )

            print()

        print(

            "\nTraining Finished."

        )