# predict.py
import torch
from pathlib import Path
from model import ExoplanetModel
import pickle

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#paths
MODEL_PATH = Path("exo_model.pth")
SCALER_PATH = Path("scaler.pkl")

#load ML model
model = ExoplanetModel(input_features=5, hidden_neurons=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Loaded trained model from {MODEL_PATH}")

#loading scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print(f"✅ Loaded scaler from {SCALER_PATH}")

def predict_exoplanet():
    try:
        # Ask for input
        koi_depth = float(input("Enter transit depth: "))
        koi_duration = float(input("Enter transit duration: "))
        koi_period = float(input("Enter orbital eriod: "))
        koi_prad = float(input("Enter planetary radius: "))
        koi_srad = float(input("Enter stellar radius: "))

        features = [[koi_depth, koi_duration, koi_period, koi_prad, koi_srad]]
        features_scaled = scaler.transform(features)

        X = torch.tensor(features_scaled, dtype=torch.float32).to(device)
        
        # Run prediction
        with torch.no_grad():
            prob = model(X).item()
            pred = 1 if prob >= 0.5 else 0

        print(f"\n🔮 Prediction: {pred} (Probability: {prob:.4f})")
        if pred == 1:
            print("➡️ Likely an exoplanet")
        else:
            print("➡️ Likely not an exoplanet")

    except ValueError:
        print("⚠️ Invalid input! Please enter numeric values.")



if __name__ == "__main__":
    while True:
        predict_exoplanet()
        again = input("\nMake another prediction? (yes/no): ").lower().strip()
        if again != "yes":
            break