import torch
import torch.nn.functional as F

from model.nidsnet import NIDSNet


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "./checkpoints/best_model.pt"


# Your 15 classes
CLASS_NAMES = [
    "Benign",                    # 0
    "Bot",                       # 1
    "DDoS",                      # 2
    "DoS GoldenEye",             # 3
    "DoS Hulk",                 # 4
    "DoS Slowhttptest",          # 5
    "DoS slowloris",             # 6
    "FTP-Patator",               # 7
    "Heartbleed",                # 8
    "Infiltration",              # 9
    "PortScan",                  # 10
    "SSH-Patator",               # 11
    "Web Attack Brute Force",    # 12
    "Web Attack Sql Injection",  # 13
    "Web Attack XSS"             # 14
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(f"Using device: {DEVICE}")

    model = NIDSNet(
        # PUT YOUR EXACT TRAINING CONFIG HERE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Handle both:
    # torch.save(model.state_dict(), ...)
    # and
    # torch.save({"model_state_dict": ...}, ...)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict(model, sample):

    model.eval()

    # Move every branch to GPU/CPU
    flow = sample["flow"].to(DEVICE)
    packet = sample["packet"].to(DEVICE)
    time = sample["time"].to(DEVICE)
    flags = sample["flags"].to(DEVICE)
    header = sample["header"].to(DEVICE)

    # Add batch dimension if necessary
    if flow.dim() == 1:
        flow = flow.unsqueeze(0)

    if packet.dim() == 1:
        packet = packet.unsqueeze(0)

    if time.dim() == 1:
        time = time.unsqueeze(0)

    if flags.dim() == 1:
        flags = flags.unsqueeze(0)

    if header.dim() == 1:
        header = header.unsqueeze(0)

    print("\nLIVE SCALED FEATURES")
    for name, value in zip(scaler.feature_names_in_, x_scaled[0]):
        print(f"{name:30s} = {value:.6f}")

    # ========================================================
    # MODEL
    # ========================================================

    output = model(
        flow,
        packet,
        time,
        flags,
        header
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = F.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    )

    confidence = probabilities[
        torch.arange(probabilities.size(0)),
        predicted_class
    ]

    return predicted_class, confidence, probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    # Load one sample from test dataset
    data = torch.load(
        "./data/processed/test.pt",
        map_location="cpu"
    )

    # First sample
    sample = {
        "flow": data["flow"][0],
        "packet": data["packet"][0],
        "time": data["time"][0],
        "flags": data["flags"][0],
        "header": data["header"][0],
    }

    predicted_class, confidence, probabilities = predict(
        model,
        sample
    )

    class_id = predicted_class.item()
    conf = confidence.item()

    print("\n==============================")
    print("       NIDS INFERENCE")
    print("==============================")

    print(f"Class ID   : {class_id}")
    print(f"Prediction : {CLASS_NAMES[class_id]}")
    print(f"Confidence : {conf * 100:.2f}%")

    print("\nClass probabilities:")

    for i, probability in enumerate(probabilities[0]):
        print(
            f"{i:2d} | "
            f"{CLASS_NAMES[i]:25s} | "
            f"{probability.item() * 100:8.4f}%"
        )


if __name__ == "__main__":
    main()