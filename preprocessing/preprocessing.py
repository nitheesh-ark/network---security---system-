import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from .feature_groups import FEATURE_GROUPS


class DataPreprocessor:

    def __init__(self, raw_data_path, output_path, test_size = 0.2, val_size =0.1, random_state = 42):

        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.X = None
        self.Y = None
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        
        self.y_train = None
        self.Y_val = None
        self.Y_test = None

        self.X_temp = None
        self.y_temp = None

        self.train_groups = {}
        self.val_groups = {}
        self.test_groups = {}

    def load_data(self):

        print("loading Dataset ...")

        parquet_files = sorted(self.raw_data_path.glob("*.parquet"))

        if len(parquet_files) == 0:
            raise FileNotFoundError("no files found")
        
        dfs = []

        for file in parquet_files:

            print(f"{file.name} Loading ...")

            dfs.append(pd.read_parquet(file))
        
        self.data = pd.concat(dfs, ignore_index = True)
        print()
        print("Dataset loaded")
        print(f"shape = {self.data.shape}")

        return self
    

    def clean_data(self):

        print("cleaning Dataset ...")
        initial_rows = len(self.data)

        #remove duplicate
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            self.data.drop_duplicates(inplace = True)
        print(f"duplicates removed : {duplicate_count}")

        #inf to nans
        self.data.replace(
            [np.inf, -np.inf], 
            np.nan, 
            inplace = True
        )

        #NaNs 
        nan_rows = self.data.isna().any(axis = 1).sum()
        if nan_rows > 0:
            self.data.dropna(inplace = True)
        print("nans removed ", nan_rows)

        #reset index
        self.data.reset_index(drop = True, inplace = True)

        final_rows = len(self.data)

        print(f"Rows before {initial_rows}")       
        print(f"Rows after {final_rows}") 
        print()

        return self


    def encode_labels(self):
        print("encoding labels")

        self.data["Label"] = self.label_encoder.fit_transform(
            self.data["Label"]
        )

        print("Label mapping")

        for index, label in enumerate(self.label_encoder.classes_):
            print(f"{label:<25} -> {index}")
        
        encoder_path = self.output_path / "label_encoder.pkl"

        joblib.dump(self.label_encoder, encoder_path)
        print(f"saved label encoder {encoder_path}")
        print("\nAfter encode_labels()")
        print(self.data["Label"].head())
        print("dtype:", self.data["Label"].dtype)
        print()
        


        mapping = {
            label: int(index)
            for index, label in enumerate(self.label_encoder.classes_)
        }

        with open(self.output_path / "label_mapping.json", "w") as f:
            json.dump(mapping, f, indent=4)

        return self


    def split_features_labels(self):

        print("=" * 60)
        print("Splitting Features and Labels")
        print("=" * 60)

        if self.data["Label"].dtype == object:
            raise ValueError(
                "Labels are still strings. Did you call encode_labels()?"
            )

        self.X = self.data.drop(columns=["Label"])
        self.Y = self.data["Label"]

        print(f"Feature Shape : {self.X.shape}")
        print(f"Label Shape   : {self.Y.shape}")
        print(f"Number of Features : {self.X.shape[1]}")
        print("\nAfter split_features_labels()")
        print(self.Y.head())
        print("dtype:", self.Y.dtype)

        return self


    def split_dataset(self):

        print("=" * 60)
        print("Splitting Dataset")
        print("=" * 60)

        # ----------------------------------------
        # First Split
        # ----------------------------------------
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(
            self.X,
            self.Y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.Y,
            shuffle=True,
        )

        # ----------------------------------------
        # Second Split
        # ----------------------------------------
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_temp,
            self.y_temp,
            test_size=0.5,
            random_state=self.random_state,
            stratify=self.y_temp,
            shuffle=True,
        )

        print(f"Training Samples   : {len(self.X_train):,}")
        print(f"Validation Samples : {len(self.X_val):,}")
        print(f"Testing Samples    : {len(self.X_test):,}")

        print()
        print("\nAfter split_dataset()")
        print(self.y_train.head())
        print("dtype:", self.y_train.dtype)

        return self

    def fit_scaler(self):

        print("=" * 60)
        print("Scaling Features")
        print("=" * 60)

        # ----------------------------
        # Fit only on training data
        # ----------------------------
        self.scaler.fit(self.X_train)

        # ----------------------------
        # Transform all splits
        # ----------------------------

        columns = self.X.columns

        self.X_train = pd.DataFrame(
            self.scaler.transform(self.X_train),
            columns=columns,
        )

        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=columns,
        )

        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=columns,
        )

        # ----------------------------
        # Save scaler
        # ----------------------------
        scaler_path = self.output_path / "scaler.pkl"

        joblib.dump(self.scaler, scaler_path)

        print(f"Scaler saved to : {scaler_path}")
        print("Training Mean :", self.X_train.mean())
        print("Training Std  :", self.X_train.std())
        print()

        return self

    def create_feature_groups(self):

        print("=" * 60)
        print("Creating Feature Groups")
        print("=" * 60)

        # Clear previous groups
        self.train_groups = {}
        self.val_groups = {}
        self.test_groups = {}

        for group_name, feature_list in FEATURE_GROUPS.items():

            missing = [f for f in feature_list if f not in self.X_train.columns]

            if missing:
                raise ValueError(
                    f"Missing features in '{group_name}': {missing}"
                )

            self.train_groups[group_name] = self.X_train[feature_list].copy()
            self.val_groups[group_name] = self.X_val[feature_list].copy()
            self.test_groups[group_name] = self.X_test[feature_list].copy()

            print(f"{group_name:<10}: {len(feature_list)} features")

        print()

        return self
    

    def save_processed_data(self):

        print("=" * 60)
        print("Saving Processed Dataset")
        print("=" * 60)

        train_data = {}
        val_data = {}
        test_data = {}

        # -----------------------------
        # Convert feature groups
        # -----------------------------
        for group in self.train_groups.keys():

            train_data[group] = torch.tensor(
                self.train_groups[group].values,
                dtype=torch.float32
            )

            val_data[group] = torch.tensor(
                self.val_groups[group].values,
                dtype=torch.float32
            )

            test_data[group] = torch.tensor(
                self.test_groups[group].values,
                dtype=torch.float32
            )

        # -----------------------------
        # Labels
        # -----------------------------
        train_data["labels"] = torch.tensor(
            self.y_train.values,
            dtype=torch.long
        )

        val_data["labels"] = torch.tensor(
            self.y_val.values,
            dtype=torch.long
        )

        test_data["labels"] = torch.tensor(
            self.y_test.values,
            dtype=torch.long
        )

        # -----------------------------
        # Save
        # -----------------------------
        torch.save(
            train_data,
            self.output_path / "train.pt"
        )

        torch.save(
            val_data,
            self.output_path / "val.pt"
        )

        torch.save(
            test_data,
            self.output_path / "test.pt"
        )

        print("train.pt saved")
        print("val.pt saved")
        print("test.pt saved")

        print()

        return self