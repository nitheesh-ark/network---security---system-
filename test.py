import pandas as pd 
print("imported -- ")


df = pd.read_parquet("./data/raw/Benign-Monday-no-metadata.parquet")
print(df.head())
print(df.shape)
print()

print("extrnal module - preprocessing pipline")

from preprocessing.preprocessing import DataPreprocessor

data_processor = DataPreprocessor(
    raw_data_path = "./data/raw",
    output_path = "./data/processed"
)


import torch


train = torch.load("data/processed/train.pt")

print(train.keys())

for key, value in train.items():
    if hasattr(value, "shape"):
        print(f"{key:<10} {value.shape} {value.dtype}")




from model.nidsnet import NIDSNet
model = NIDSNet()

batch_size = 8

flow = torch.randn(batch_size, 18)

packet = torch.randn(batch_size, 12)

time = torch.randn(batch_size, 6)

flags = torch.randn(batch_size, 9)

header = torch.randn(batch_size, 14)

output = model(
    flow,
    packet,
    time,
    flags,
    header
)

print(output)
print(output.shape)

predictions = torch.argmax(output, dim=1)

print(predictions)