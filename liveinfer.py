import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify

from model.nidsnet import NIDSNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "./checkpoints/best_model.pt"
SCALER_PATH = "./data/processed/scaler.pkl"

HOST = "127.0.0.1"
PORT = 8080

BENIGN_CLASS = 0


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
# TRAINING FEATURE GROUPS
#
# MUST MATCH TRAINING EXACTLY
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


# ============================================================
# FEATURE GROUPS
# ============================================================

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
# SANITY CHECKS
# ============================================================

assert len(FLOW_FEATURES) == 10
assert len(PACKET_FEATURES) == 18
assert len(TIME_FEATURES) == 22
assert len(FLAG_FEATURES) == 12
assert len(HEADER_FEATURES) == 15

assert len(ALL_FEATURES) == 77


# ============================================================
# CICFLOWMETER -> TRAINING FEATURE MAPPING
#
# IMPORTANT:
# Every training feature MUST appear here exactly once.
# ============================================================

FEATURE_MAP = {

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    "Flow Duration":
        "flow_duration",

    "Flow Bytes/s":
        "flow_byts_s",

    "Flow Packets/s":
        "flow_pkts_s",

    "Total Fwd Packets":
        "tot_fwd_pkts",

    "Total Backward Packets":
        "tot_bwd_pkts",

    "Subflow Fwd Packets":
        "subflow_fwd_pkts",

    "Subflow Bwd Packets":
        "subflow_bwd_pkts",

    "Subflow Fwd Bytes":
        "subflow_fwd_byts",

    "Subflow Bwd Bytes":
        "subflow_bwd_byts",

    "Down/Up Ratio":
        "down_up_ratio",


    # --------------------------------------------------------
    # PACKET
    # --------------------------------------------------------

    "Packet Length Min":
        "pkt_len_min",

    "Packet Length Max":
        "pkt_len_max",

    "Packet Length Mean":
        "pkt_len_mean",

    "Packet Length Std":
        "pkt_len_std",

    "Packet Length Variance":
        "pkt_len_var",

    "Avg Packet Size":
        "pkt_size_avg",

    "Fwd Packet Length Max":
        "fwd_pkt_len_max",

    "Fwd Packet Length Min":
        "fwd_pkt_len_min",

    "Fwd Packet Length Mean":
        "fwd_pkt_len_mean",

    "Fwd Packet Length Std":
        "fwd_pkt_len_std",

    "Bwd Packet Length Max":
        "bwd_pkt_len_max",

    "Bwd Packet Length Min":
        "bwd_pkt_len_min",

    "Bwd Packet Length Mean":
        "bwd_pkt_len_mean",

    "Bwd Packet Length Std":
        "bwd_pkt_len_std",

    "Fwd Packets Length Total":
        "totlen_fwd_pkts",

    "Bwd Packets Length Total":
        "totlen_bwd_pkts",

    "Avg Fwd Segment Size":
        "fwd_seg_size_avg",

    "Avg Bwd Segment Size":
        "bwd_seg_size_avg",


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    "Flow IAT Mean":
        "flow_iat_mean",

    "Flow IAT Std":
        "flow_iat_std",

    "Flow IAT Max":
        "flow_iat_max",

    "Flow IAT Min":
        "flow_iat_min",

    "Fwd IAT Total":
        "fwd_iat_tot",

    "Fwd IAT Mean":
        "fwd_iat_mean",

    "Fwd IAT Std":
        "fwd_iat_std",

    "Fwd IAT Max":
        "fwd_iat_max",

    "Fwd IAT Min":
        "fwd_iat_min",

    "Bwd IAT Total":
        "bwd_iat_tot",

    "Bwd IAT Mean":
        "bwd_iat_mean",

    "Bwd IAT Std":
        "bwd_iat_std",

    "Bwd IAT Max":
        "bwd_iat_max",

    "Bwd IAT Min":
        "bwd_iat_min",

    "Active Mean":
        "active_mean",

    "Active Std":
        "active_std",

    "Active Max":
        "active_max",

    "Active Min":
        "active_min",

    "Idle Mean":
        "idle_mean",

    "Idle Std":
        "idle_std",

    "Idle Max":
        "idle_max",

    "Idle Min":
        "idle_min",


    # --------------------------------------------------------
    # FLAGS
    # --------------------------------------------------------

    "FIN Flag Count":
        "fin_flag_cnt",

    "SYN Flag Count":
        "syn_flag_cnt",

    "RST Flag Count":
        "rst_flag_cnt",

    "PSH Flag Count":
        "psh_flag_cnt",

    "ACK Flag Count":
        "ack_flag_cnt",

    "URG Flag Count":
        "urg_flag_cnt",

    # IMPORTANT:
    # This was missing in your previous code.
    #
    # CICFlowMeter:
    #     cwr_flag_count
    #
    # Training:
    #     CWE Flag Count
    #
    "CWE Flag Count":
        "cwr_flag_count",

    "ECE Flag Count":
        "ece_flag_cnt",

    "Fwd PSH Flags":
        "fwd_psh_flags",

    "Bwd PSH Flags":
        "bwd_psh_flags",

    "Fwd URG Flags":
        "fwd_urg_flags",

    "Bwd URG Flags":
        "bwd_urg_flags",


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    "Protocol":
        "protocol",

    "Fwd Header Length":
        "fwd_header_len",

    "Bwd Header Length":
        "bwd_header_len",

    "Fwd Packets/s":
        "fwd_pkts_s",

    "Bwd Packets/s":
        "bwd_pkts_s",

    "Init Fwd Win Bytes":
        "init_fwd_win_byts",

    "Init Bwd Win Bytes":
        "init_bwd_win_byts",

    "Fwd Act Data Packets":
        "fwd_act_data_pkts",

    "Fwd Seg Size Min":
        "fwd_seg_size_min",

    "Fwd Avg Bytes/Bulk":
        "fwd_byts_b_avg",

    "Fwd Avg Packets/Bulk":
        "fwd_pkts_b_avg",

    "Fwd Avg Bulk Rate":
        "fwd_blk_rate_avg",

    "Bwd Avg Bytes/Bulk":
        "bwd_byts_b_avg",

    "Bwd Avg Packets/Bulk":
        "bwd_pkts_b_avg",

    "Bwd Avg Bulk Rate":
        "bwd_blk_rate_avg",
}


# ============================================================
# VERIFY MAPPING
# ============================================================

assert len(FEATURE_MAP) == 77

missing_mapping = [
    feature
    for feature in ALL_FEATURES
    if feature not in FEATURE_MAP
]

if missing_mapping:
    raise RuntimeError(
        "FEATURE_MAP is missing:\n"
        + "\n".join(missing_mapping)
    )

print(
    f"Feature mapping verified: "
    f"{len(FEATURE_MAP)}/77"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("                    NIDSNet LIVE")
print("=" * 70)

print(f"Device : {DEVICE}")

if DEVICE.type == "cuda":
    print(
        "GPU    :",
        torch.cuda.get_device_name(0)
    )

print("Loading model...")

model = NIDSNet()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)
model.eval()


print("Model  : NIDSNet")
print("Status : LOADED")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = None


if os.path.exists(SCALER_PATH):

    print(
        f"Scaler : {SCALER_PATH}"
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print("Scaler : LOADED")

else:

    print()
    print("WARNING")
    print("-" * 70)
    print(
        f"Scaler not found:\n{SCALER_PATH}"
    )
    print()
    print(
        "Running WITHOUT scaling."
    )
    print(
        "Predictions may be unreliable if "
        "your model was trained with scaling."
    )
    print("-" * 70)


# ============================================================
# PRINT MODEL INPUT SHAPES
# ============================================================

print()
print("Expected branch dimensions:")
print(
    f"Flow   : {len(FLOW_FEATURES)}"
)
print(
    f"Packet : {len(PACKET_FEATURES)}"
)
print(
    f"Time   : {len(TIME_FEATURES)}"
)
print(
    f"Flags  : {len(FLAG_FEATURES)}"
)
print(
    f"Header : {len(HEADER_FEATURES)}"
)
print(
    f"TOTAL  : {len(ALL_FEATURES)}"
)

print("=" * 70)
print()


# ============================================================
# HELPER
# ============================================================

def get_source_ip(data):

    return (
        data.get("src_ip")
        or data.get("Src IP")
        or data.get("Source IP")
        or "?"
    )


def get_destination_ip(data):

    return (
        data.get("dst_ip")
        or data.get("Dst IP")
        or data.get("Destination IP")
        or "?"
    )


# ============================================================
# BUILD FEATURE VECTOR
# ============================================================

def build_feature_dataframe(data):

    values = {}

    missing = []

    for model_feature in ALL_FEATURES:

        live_name = FEATURE_MAP[model_feature]

        if live_name not in data:

            missing.append(
                f"{model_feature} -> {live_name}"
            )

            continue

        try:

            values[model_feature] = float(
                data[live_name]
            )

        except (TypeError, ValueError):

            values[model_feature] = 0.0


    if missing:

        raise ValueError(
            "Missing live features:\n"
            + "\n".join(missing)
        )


    # --------------------------------------------------------
    # EXACT TRAINING ORDER
    # --------------------------------------------------------

    df = pd.DataFrame(
        [[
            values[feature]
            for feature in ALL_FEATURES
        ]],
        columns=ALL_FEATURES
    )


    # --------------------------------------------------------
    # CLEAN NUMBERS
    # --------------------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.fillna(
        0,
        inplace=True
    )


    return df


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_live_flow(data):

    df = build_feature_dataframe(
        data
    )


    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    if scaler is not None:

        scaled = scaler.transform(
            df.values
        )

        df = pd.DataFrame(
            scaled,
            columns=ALL_FEATURES
        )


    # --------------------------------------------------------
    # FIVE BRANCHES
    # --------------------------------------------------------

    flow_x = df[
        FLOW_FEATURES
    ].values

    packet_x = df[
        PACKET_FEATURES
    ].values

    time_x = df[
        TIME_FEATURES
    ].values

    flags_x = df[
        FLAG_FEATURES
    ].values

    header_x = df[
        HEADER_FEATURES
    ].values


    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    tensors = {

        "flow": torch.tensor(
            flow_x,
            dtype=torch.float32,
            device=DEVICE
        ),

        "packet": torch.tensor(
            packet_x,
            dtype=torch.float32,
            device=DEVICE
        ),

        "time": torch.tensor(
            time_x,
            dtype=torch.float32,
            device=DEVICE
        ),

        "flags": torch.tensor(
            flags_x,
            dtype=torch.float32,
            device=DEVICE
        ),

        "header": torch.tensor(
            header_x,
            dtype=torch.float32,
            device=DEVICE
        ),
    }


    # --------------------------------------------------------
    # HARD SAFETY CHECK
    # --------------------------------------------------------

    expected_shapes = {
        "flow": 10,
        "packet": 18,
        "time": 22,
        "flags": 12,
        "header": 15,
    }


    for name, expected in expected_shapes.items():

        actual = tensors[name].shape[1]

        if actual != expected:

            raise RuntimeError(
                f"{name} branch has "
                f"{actual} features. "
                f"Expected {expected}."
            )


    return tensors


# ============================================================
# RUN MODEL
# ============================================================

def run_inference(data):

    tensors = preprocess_live_flow(
        data
    )


    flow = tensors["flow"]
    packet = tensors["packet"]
    time_features = tensors["time"]
    flags = tensors["flags"]
    header = tensors["header"]


    # --------------------------------------------------------
    # DEBUG SHAPES
    # --------------------------------------------------------

    print()
    print(
        f"INPUT SHAPES | "
        f"flow={tuple(flow.shape)} "
        f"packet={tuple(packet.shape)} "
        f"time={tuple(time_features.shape)} "
        f"flags={tuple(flags.shape)} "
        f"header={tuple(header.shape)}"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            flow,
            packet,
            time_features,
            flags,
            header
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    class_id = int(
        prediction.item()
    )

    confidence_value = float(
        confidence.item()
    )


    if class_id >= len(CLASS_NAMES):

        class_name = f"Class {class_id}"

    else:

        class_name = CLASS_NAMES[
            class_id
        ]


    probability_list = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )


    return (
        class_id,
        class_name,
        confidence_value,
        probability_list
    )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def print_prediction(
    data,
    class_id,
    class_name,
    confidence
):

    now = datetime.now().strftime(
        "%H:%M:%S"
    )

    source = get_source_ip(
        data
    )

    destination = get_destination_ip(
        data
    )


    print()
    print(
        f"{now}  "
        f"{str(source):<18} "
        f"{str(destination):<18} "
        f"{class_name:<25} "
        f"{confidence * 100:6.2f}%"
        + (
            " 🚨"
            if class_id != BENIGN_CLASS
            else ""
        )
    )


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict_endpoint():

    try:

        data = request.get_json(
            force=True
        )


        if not data:

            return jsonify({
                "error":
                    "Empty flow"
            }), 400


        # ----------------------------------------------------
        # RUN INFERENCE
        # ----------------------------------------------------

        (
            class_id,
            class_name,
            confidence,
            probabilities
        ) = run_inference(
            data
        )


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print_prediction(
            data,
            class_id,
            class_name,
            confidence
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "status":
                "success",

            "class_id":
                class_id,

            "prediction":
                class_name,

            "confidence":
                confidence,

            "confidence_percent":
                confidence * 100,

            "source":
                get_source_ip(data),

            "destination":
                get_destination_ip(data),

            "probabilities":
                probabilities,

        }), 200


    except Exception as e:

        print()
        print(
            "=" * 70
        )
        print(
            "[INFERENCE ERROR]"
        )
        print(
            "=" * 70
        )
        print(
            str(e)
        )

        import traceback

        traceback.print_exc()

        print(
            "=" * 70
        )


        return jsonify({

            "status":
                "error",

            "error":
                str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "running",

        "model":
            "NIDSNet",

        "device":
            str(DEVICE),

        "features":
            len(ALL_FEATURES),

        "feature_groups": {

            "flow":
                len(FLOW_FEATURES),

            "packet":
                len(PACKET_FEATURES),

            "time":
                len(TIME_FEATURES),

            "flags":
                len(FLAG_FEATURES),

            "header":
                len(HEADER_FEATURES),
        }

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )
    print(
        "                 NIDSNet LIVE SERVER"
    )
    print(
        "=" * 70
    )

    print(
        f"Device    : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU       :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Model     : NIDSNet"
    )

    print(
        "Features  : 77"
    )

    print(
        "Flow      : 10"
    )

    print(
        "Packet    : 18"
    )

    print(
        "Time      : 22"
    )

    print(
        "Flags     : 12"
    )

    print(
        "Header    : 15"
    )

    print(
        "Status    : MONITORING"
    )

    print()
    print(
        "Inference endpoint:"
    )

    print(
        f"http://{HOST}:{PORT}/predict"
    )

    print()
    print(
        "Waiting for live flows..."
    )

    print(
        "=" * 70
    )
    print()


    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True
    )