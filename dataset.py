"""
dataset.py

PyTorch Dataset for NIDSNet
"""

import torch
from torch.utils.data import Dataset


class NIDSDataset(Dataset):

    def __init__(self, pt_file):

        data = torch.load(pt_file)

        self.flow = data["flow"]
        self.packet = data["packet"]
        self.time = data["time"]
        self.flags = data["flags"]
        self.header = data["header"]
        self.labels = data["labels"]

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return {
            "flow": self.flow[idx],
            "packet": self.packet[idx],
            "time": self.time[idx],
            "flags": self.flags[idx],
            "header": self.header[idx],
            "label": self.labels[idx]
        }

if __name__ == "__main__":

    # Example usage
    dataset = NIDSDataset("./data/processed/train.pt")
    print("Dataset length:", len(dataset))

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")