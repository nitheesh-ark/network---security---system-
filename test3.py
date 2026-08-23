"""
test.py

Evaluate the trained NIDSNet model on the test dataset.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import NIDSDataset
from model.nidsnet import NIDSNet


def main():

    ##########################################
    # Device
    ##########################################

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ##########################################
    # Load Test Dataset
    ##########################################

    test_dataset = NIDSDataset("./data/processed/test.pt")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    ##########################################
    # Load Model
    ##########################################

    model = NIDSNet().to(device)

    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    ##########################################
    # Testing
    ##########################################

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in test_loader:

            flow = batch["flow"].to(device)
            packet = batch["packet"].to(device)
            time = batch["time"].to(device)
            flags = batch["flags"].to(device)
            header = batch["header"].to(device)

            labels = batch["label"].to(device)

            logits = model(
                flow,
                packet,
                time,
                flags,
                header
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    ##########################################
    # Metrics
    ##########################################

    accuracy = accuracy_score(
        targets,
        predictions
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            targets,
            predictions,
            digits=4
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            targets,
            predictions
        )
    )


if __name__ == "__main__":
    main()