import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pickle

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

train_size = int(0.7 * len(dataset)) # Takes 70% of the data set for training
val_size = int(0.15 * len(dataset)) # This takes 15% for  validation
test_size = len(dataset)- train_size - val_size # Remaining goes to testing

# Splitting the data to avoid bias
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Goes through 34 batches of data at a time
batch_size = 34
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# The neural network
class ExoPlanetClassifier(nn.Module):
    def __init__(self, input_size):
        super(ExoPlanetClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

input_size = len(features)
model = ExoPlanetClassifier(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")

print("\nEvaluating on test set...")
model.eval()
test_correct = 0
test_total = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features).squeeze()
        predictions = (outputs > 0.5).float()

        test_correct += (predictions == batch_labels).sum().item()
        test_total += batch_labels.size(0)

        for pred, label in zip(predictions, batch_labels):
            if pred == 1 and label == 1:
                true_positives += 1
            elif pred == 1 and label == 0:
                false_positives += 1
            elif pred == 0 and label == 0:
                true_negatives += 1
            elif pred == 0 and label == 1:
                false_negatives += 1

test_accuracy = 100 * test_correct / test_total
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"TEST SET RESULTS")
print(f"\nAccuracy: {test_accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"\nConfusion Matrix:")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")

# Save the model weights
torch.save(model.state_dict(), 'exoplanet_model.pth')
# Save the scaler so Flask can use the same normalization
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print(f"\nModel saved as 'exoplanet_model.pth'")
print(f"Scaler saved as 'scaler.pkl'")