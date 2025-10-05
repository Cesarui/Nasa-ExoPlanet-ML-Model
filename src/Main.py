from pyexpat import features

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Loads the csv correctly since for some odd reason it wouldn't regularly work
df = pd.read_csv(
    "/Users/cesarpimentel/Desktop/PythonProjects/NasaExoPlanetMLModel/Data/cumulative_2025.10.05_12.30.33.csv",
    skiprows=52,
    comment='#',
    on_bad_lines='skip',
    low_memory=False
)

print(f"Initial shape: {df.shape}")

# All the features we want to use

features = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad", # planet radius
    "koi_srad", # solar radius
]

# Updates the data frame with only the columns we need and drops rows with NaN values
df = df[features + ["koi_disposition"]]
df = df.dropna()

df["label"] = df["koi_disposition"].apply(lambda x: 1 if x == "CONFIRMED" else 0)

df = df.drop(columns=["koi_disposition"])

print(f"After cleaning: {df.shape}") # All the other columns are gone since we won't use them.

print(f"Label distribution:\n{df['label'].value_counts()}") # Shows the amounts # of CONFIRMED as 1 and not confirmed as 0

# The class that handles the data since pytorch needs the data in a certain way
class ExoplanetDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X = df[features].values
y = df["label"].values

scaler = StandardScaler() # Scalar object
X_scaled = scaler.fit_transform(X) # Transforms(normalizes) the data with some dark magic so it's all in similar scale.

dataset = ExoplanetDataset(X_scaled, y) # Creates the data set with the normalized X and it's labels





