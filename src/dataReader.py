#this is pranshu's branch
# this reads the csv file, and outputs the important attributes needed to determine if the object is a exoplanet
import pandas as pd
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
'''
file = "C:/Users/prans/Desktop/Nasa-Exoplanet-ML-Model/test.csv"

df = pd.read_csv(file, comment="#")


#this is the og data Reader
# 
# wanted = ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade","pl_insol", "pl_eqt", "st_tmag", "st_teff","st_logg", "st_rad", "tfopwg_disp"]
df = df[wanted]

for index, row in df.iterrows():
    period = row["pl_orbper"]
    duration = row["pl_trandurh"]
    depth = row["pl_trandep"]
    radius = row["pl_rade"]
    insol = row["pl_insol"]
    eqt = row["pl_eqt"]
    tmag = row["st_tmag"]
    teff = row["st_teff"]
    logg = row["st_logg"]
    star_rad = row["st_rad"]
    label = row["tfopwg_disp"]

    print(f"Row {index} | Period: {period} d | Duration: {duration} h | Depth: {depth} | Radius: {radius} R_earth | Insol: {insol} | EqT: {eqt} K | Tmag: {tmag} | Teff: {teff} K | logg: {logg} | Star Radius: {star_rad} R_sun | Label: {label}")
'''
#i am implementing Ky's check 1 under here

# === Load CSV ===
from pathlib import Path

# Build a safe path to the CSV file using pathlib. This avoids Python interpreting backslashes as
# escape sequences and makes the code portable across OSes. If you prefer an absolute path,
# replace project_root / 'test.csv' with Path(r"C:\Users\prans\Desktop\Nasa-ExoPlanet-ML-Model\test.csv").
project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / 'test.csv'

if not csv_path.exists():
    # Try a fallback: user's workspace root where this script likely runs
    csv_path = Path('test.csv')

if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found at {csv_path}. Update the path to your dataset.")

df = pd.read_csv(csv_path, comment="#")

# === Define your math ===

def radius_from_depth_ppm(depth_ppm, st_rad):
    # Rp = R * sqrt(depth / 1e6), depth in ppm, R in solar radii
    if depth_ppm <= 0 or st_rad <= 0:
        return np.nan
    return st_rad * math.sqrt(depth_ppm / 1e6)

def semi_major_a_AU(P_days, R_star):
    # ((P / 365.25)^(2/3)) * (R^0.8)
    if P_days <= 0 or R_star <= 0:
        return np.nan
    return ((P_days / 365.25) ** (2/3)) * (R_star ** 0.8)

def expected_duration_hours(P_days, R_star):
    a_AU = semi_major_a_AU(P_days, R_star)
    if a_AU is np.nan or R_star <= 0:
        return np.nan
    x = (a_AU * 215.032) / R_star   # a/R
    if x <= 0:
        return np.nan
    y = (P_days / math.pi) * (1.0 / x) * 24.0
    return y

label_map = {
    "CP": 1,           # Confirmed Planet
    "CONFIRMED": 1,
    "PC": -1,          # Planet Candidate (optional, you could drop these instead)
    "FP": 0,           # False Positive
    "FALSE POSITIVE": 0
}

rows = []
for _, r in df.iterrows():
    depth = r.get("pl_trandep", np.nan)     # transit depth [ppm]
    dur = r.get("pl_trandurh", np.nan)      # observed duration [h]
    P = r.get("pl_orbper", np.nan)          # orbital period [days]
    R = r.get("st_rad", np.nan)             # stellar radius [Rsun]
    label_text = str(r.get("tfopwg_disp", "")).upper().strip()
    label = label_map.get(label_text, np.nan)  # translate into number

    rp_rsun = radius_from_depth_ppm(depth, R)
    a_AU = semi_major_a_AU(P, R)
    exp_dur = expected_duration_hours(P, R)

    ratio = dur / exp_dur if (exp_dur and exp_dur > 0) else np.nan

    rows.append({
        "pl_trandep": depth,
        "pl_trandurh": dur,
        "pl_orbper": P,
        "st_rad": R,
        "rp_rsun": rp_rsun,           # from check #1
        "a_AU": a_AU,
        "expected_dur": exp_dur,
        "ratio": ratio,               # from check #2
        "label": label                # 1=planet, 0=not, -1=candidate
    })

feat_df = pd.DataFrame(rows)

# === Inspect first few rows ===
print(feat_df.head(10))

    #just testing 

'''if period == 2.1713484:
        print("Value can be drawn.")
    else:
        print("Ignore this.")'''
'''
    #this is the model so far

    class Model(nn.Module):
    #input layer 11 inputs
    def __init__(self, input_features=11, hlayer1= 8, hlayer2 = 9, output_features = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hlayer1)
        self.fc2 = nn.Linear(hlayer1, hlayer2)
        self.out = nn.Linear(hlayer2, output_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

#random seed for randominzation
torch.manual_seed(41)

mlModel = Model()

'''

