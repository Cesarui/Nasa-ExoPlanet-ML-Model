# nasa_exo_ml/train_model.py

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from dataReader import load_exoplanet_data
from model import ExoplanetModel

from sklearn.preprocessing import StandardScaler
import pickle

#setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

#Paths
MODEL_PATH = Path("exo_model.pth")
CSV_PATH = "cumulative_2025.10.04_07.30.24.csv"

#Load and prepare data
data = load_exoplanet_data(CSV_PATH)
X = data[["koi_depth", "koi_duration", "koi_period", "koi_prad", "koi_srad"]].values
y = data["koi_disposition"].values

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for Flask later
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

#Initialize model
model = ExoplanetModel(input_features=5, hidden_neurons=128).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Loading saved model
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded saved model from {MODEL_PATH}")
else:
    print("🚀 No saved model found — training a new one...")

#Training loop
epochs = 25
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct_train, total_train = 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #training accuracy
        predicted = (preds >= 0.5).float()
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    #Validation
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            predicted = (preds >= 0.5).float()
            correct_val += (predicted == y_batch).sum().item()
            total_val += y_batch.size(0)

    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
print(f"\n💾 Model saved to {MODEL_PATH}")

#Final Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        predicted = (preds >= 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"\n✅ Overall Test Accuracy: {accuracy*100:.2f}%")