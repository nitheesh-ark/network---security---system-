# NIDSNet 🛡️

NIDSNet is a **Network Intrusion Detection System** built using deep learning and PyTorch.

It is designed to detect and classify different types of network attacks from network traffic data.

## Features

* Deep learning based intrusion detection
* Multi-class attack classification
* PyTorch implementation
* GPU support using CUDA
* Processes network flow features
* Supports live inference using Flask
* Achieves **99.33% test accuracy**

## Model Architecture

NIDSNet uses multiple branches to process different types of network features:

```text
Network Traffic
      ↓
 ┌────┬────┬────┬────┬────┐
 ↓    ↓    ↓    ↓    ↓
Flow Packet Time Flags Header
 └────┴────┴────┴────┴────┘
              ↓
       Feature Fusion
              ↓
      Classification Head
              ↓
       Attack Prediction
```

## Dataset

* Original samples: **2,313,810**
* Samples after cleaning: **2,231,806**
* Features: **78**
* Classes: **15**

## Results

| Metric              |      Score |
| ------------------- | ---------: |
| Training Accuracy   |     99.21% |
| Validation Accuracy |     99.29% |
| Test Accuracy       | **99.33%** |

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Flask
* Scapy
* CICFlowMeter

## Live Inference

The trained model can be connected to a Flask API for network traffic prediction.

```text
Network Traffic
      ↓
CICFlowMeter / Scapy
      ↓
Feature Extraction
      ↓
NIDSNet
      ↓
Attack Prediction
```

## Project Status

🚧 **Under Development**

The project is currently being improved for real-time network monitoring and deployment.

## Author

**Nitheesh Kumar**

Deep Learning | Network Security | PyTorch
