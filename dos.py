import os
import joblib
import pandas as pd
import torch

from model.nidsnet import NIDSNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_PATH = r"./data/raw/DDoS-Friday-no-metadata.parquet"
CHECKPOINT_PATH = r"./checkpoints/best_model.pt"
SCALER_PATH = r"./data/processed/scaler.pkl"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Benign",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack Brute Force",
    "Web Attack Sql Injection",
    "Web Attack XSS",
]


# ============================================================
# FEATURE GROUPS
# MUST MATCH TRAINING
# ============================================================

FLOW_FEATURES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Down/Up Ratio",
]


PACKET_FEATURES = [
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Avg Packet Size",

    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",

    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",

    "Fwd Packets Length Total",
    "Bwd Packets Length Total",

    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]


TIME_FEATURES = [
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",

    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",

    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",

    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",

    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


FLAG_FEATURES = [
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",

    "Fwd PSH Flags",
    "Bwd PSH Flags",

    "Fwd URG Flags",
    "Bwd URG Flags",
]


HEADER_FEATURES = [
    "Protocol",
    "Fwd Header Length",
    "Bwd Header Length",

    "Fwd Packets/s",
    "Bwd Packets/s",

    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",

    "Fwd Act Data Packets",
    "Fwd Seg Size Min",

    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",

    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


FEATURE_GROUPS = {
    "flow": FLOW_FEATURES,
    "packet": PACKET_FEATURES,
    "time": TIME_FEATURES,
    "flags": FLAG_FEATURES,
    "header": HEADER_FEATURES,
}


ALL_FEATURES = (
    FLOW_FEATURES
    + PACKET_FEATURES
    + TIME_FEATURES
    + FLAG_FEATURES
    + HEADER_FEATURES
)


# ============================================================
# FEATURE COUNT CHECK
# ============================================================

print("=" * 70)
print("NIDSNet DDoS INFERENCE TEST")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

print("\nFeature groups:")

for name, features in FEATURE_GROUPS.items():
    print(f"{name:<10}: {len(features)}")

print(f"{'TOTAL':<10}: {len(ALL_FEATURES)}")

if len(ALL_FEATURES) != 77:
    raise RuntimeError(
        f"Expected 77 features, got {len(ALL_FEATURES)}"
    )


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    DATASET_PATH,
    CHECKPOINT_PATH,
    SCALER_PATH,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("Loading DDoS dataset")
print("=" * 70)

df = pd.read_parquet(DATASET_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURE CHECK
# ============================================================

missing = [
    feature
    for feature in ALL_FEATURES
    if feature not in df.columns
]

if missing:

    print("\nMissing features:")

    for feature in missing:
        print(f" - {feature}")

    raise RuntimeError(
        f"Dataset is missing {len(missing)} features."
    )

print(f"\nRequired features: {len(ALL_FEATURES)}")
print("Feature check: OK")


# ============================================================
# LABEL CHECK
# ============================================================

if "Label" not in df.columns:
    raise RuntimeError(
        "Dataset does not contain 'Label' column."
    )

print("\nAvailable labels:")
print(df["Label"].value_counts())


# ============================================================
# SELECT A DDOS SAMPLE
# ============================================================

ddos_df = df[
    df["Label"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "ddos"
].copy()


if len(ddos_df) == 0:
    raise RuntimeError(
        "No DDoS samples found."
    )


print(
    f"\nDDoS samples found: {len(ddos_df)}"
)


# Use the first DDoS sample
sample = ddos_df.iloc[[0]].copy()

print("\nSelected sample")
print("Expected class: DDoS")


# ============================================================
# EXTRACT FEATURES
# ============================================================

x = sample[ALL_FEATURES].copy()

print("\nRaw feature shape:")
print(x.shape)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded.")


# ============================================================
# ALIGN WITH SCALER
# ============================================================

print("\nAligning features to scaler...")

if not hasattr(
    scaler,
    "feature_names_in_"
):
    raise RuntimeError(
        "Scaler does not contain "
        "'feature_names_in_'."
    )


SCALER_FEATURES = list(
    scaler.feature_names_in_
)


print(
    f"Scaler expects: "
    f"{len(SCALER_FEATURES)} features"
)


if len(SCALER_FEATURES) != 77:
    raise RuntimeError(
        f"Scaler expects "
        f"{len(SCALER_FEATURES)} features, "
        f"not 77."
    )


missing_scaler = [
    feature
    for feature in SCALER_FEATURES
    if feature not in x.columns
]


if missing_scaler:

    print("\nMissing scaler features:")

    for feature in missing_scaler:
        print(f" - {feature}")

    raise RuntimeError(
        "Input does not contain "
        "all scaler features."
    )


# IMPORTANT:
# Use exactly the order used when
# the scaler was fitted.

x = x[
    SCALER_FEATURES
].copy()


print(
    "Feature order aligned successfully."
)


# ============================================================
# SCALE
# ============================================================

print("\nScaling features...")

x_scaled_array = scaler.transform(x)


x_scaled = pd.DataFrame(
    x_scaled_array,
    columns=SCALER_FEATURES,
    index=x.index,
)


print(
    f"Scaled shape: "
    f"{x_scaled.shape}"
)


# ============================================================
# CREATE FIVE MODEL BRANCHES
# ============================================================

print("\n" + "=" * 70)
print("Creating model inputs")
print("=" * 70)


flow = torch.tensor(
    x_scaled[
        FLOW_FEATURES
    ].values,
    dtype=torch.float32,
    device=DEVICE,
)


packet = torch.tensor(
    x_scaled[
        PACKET_FEATURES
    ].values,
    dtype=torch.float32,
    device=DEVICE,
)


time_features = torch.tensor(
    x_scaled[
        TIME_FEATURES
    ].values,
    dtype=torch.float32,
    device=DEVICE,
)


flags = torch.tensor(
    x_scaled[
        FLAG_FEATURES
    ].values,
    dtype=torch.float32,
    device=DEVICE,
)


header = torch.tensor(
    x_scaled[
        HEADER_FEATURES
    ].values,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# PRINT INPUT SHAPES
# ============================================================

print("\nINPUT SHAPES")

print(
    f"flow   = {tuple(flow.shape)}"
)

print(
    f"packet = {tuple(packet.shape)}"
)

print(
    f"time   = {tuple(time_features.shape)}"
)

print(
    f"flags  = {tuple(flags.shape)}"
)

print(
    f"header = {tuple(header.shape)}"
)


# ============================================================
# VERIFY SHAPES
# ============================================================

expected = {
    "flow": 10,
    "packet": 18,
    "time": 22,
    "flags": 12,
    "header": 15,
}


actual = {
    "flow": flow.shape[1],
    "packet": packet.shape[1],
    "time": time_features.shape[1],
    "flags": flags.shape[1],
    "header": header.shape[1],
}


for name in expected:

    if actual[name] != expected[name]:

        raise RuntimeError(
            f"{name} shape mismatch: "
            f"expected {expected[name]}, "
            f"got {actual[name]}"
        )


print(
    "\nAll model input shapes are correct."
)


# ============================================================
# LOAD NIDSNET
# ============================================================

print("\n" + "=" * 70)
print("Loading NIDSNet")
print("=" * 70)


# These values match your actual NIDSNet constructor.

model = NIDSNet(
    embedding_dim=16,
    hidden_dims=[64, 32],
    fusion_hidden=[128, 64],
    num_classes=15,
    dropout=0.3,
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)


if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(
    state_dict
)

model.to(DEVICE)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# RUN INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)


with torch.no_grad():

    logits = model(
        flow,
        packet,
        time_features,
        flags,
        header,
    )


# ============================================================
# SOFTMAX
# ============================================================

probabilities = torch.softmax(
    logits,
    dim=1,
)


# ============================================================
# PREDICTION
# ============================================================

predicted_class = torch.argmax(
    probabilities,
    dim=1,
).item()


confidence = probabilities[
    0,
    predicted_class,
].item()


predicted_label = (
    CLASS_NAMES[
        predicted_class
    ]
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("NIDSNet RESULT")
print("=" * 70)

print(
    "\nExpected  : DDoS"
)

print(
    f"Predicted : {predicted_label}"
)

print(
    f"Confidence: "
    f"{confidence * 100:.2f}%"
)


# ============================================================
# ALL CLASS PROBABILITIES
# ============================================================

print("\nClass probabilities:")

for i, class_name in enumerate(
    CLASS_NAMES
):

    probability = (
        probabilities[
            0,
            i
        ].item()
    )

    print(
        f"{class_name:<30}"
        f"{probability * 100:8.4f}%"
    )


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)


if predicted_class == 2:

    print(
        "✅ DDoS DETECTED"
    )

else:

    print(
        "❌ DDoS NOT DETECTED"
    )

    print(
        f"Model predicted: "
        f"{predicted_label}"
    )


print("=" * 70)