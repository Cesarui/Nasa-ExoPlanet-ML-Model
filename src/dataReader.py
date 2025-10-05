#this is pranshu's branch
# this reads the csv file, and outputs the important attributes needed to determine if the object is a exoplanet
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

# Build a safe path to the CSV file using pathlib. This avoids Python interpreting backslashes as
# escape sequences and makes the code portable across OSes. If you prefer an absolute path,
# replace project_root / 'test.csv' with Path(r"C:\Users\prans\Desktop\Nasa-ExoPlanet-ML-Model\test.csv").
'''
if not csv_path.exists():
    # Try a fallback: user's workspace root where this script likely runs
    csv_path = Path('cumulative_2025.10.04_07.30.24.csv')

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
    "CANDIDATE": 1,           # Confirmed Planet
    "FALSE POSITIVE": 0       # False Positive
}

rows = []
for _, r in df.iterrows():
    depth = r.get("koi_depth", np.nan)     # transit depth [ppm]
    dur = r.get("koi_duration", np.nan)      # observed duration [h]
    P = r.get("koi_period", np.nan)          # orbital period [days]
    R = r.get("koi_prad", np.nan)             # stellar radius [Rsun]
    label_text = str(r.get("koi_disposition", "")).upper().strip()
    label = label_map.get(label_text, np.nan)  # translate into number

    rp_rsun = radius_from_depth_ppm(depth, R)
    a_AU = semi_major_a_AU(P, R)
    exp_dur = expected_duration_hours(P, R)

    ratio = dur / exp_dur if (exp_dur and exp_dur > 0) else np.nan

    rows.append({
        "koi_depth": depth,
        "koi_duration": dur,
        "koi_period": P,
        "koi_prad": R,
        #"rp_rsun": rp_rsun,           # from check #1
        #"a_AU": a_AU,
        #"expected_dur": exp_dur,
        #"ratio": ratio,               # from check #2
        "koi_disposition": label                # 1=planet, 0=not, -1=candidate
    })

feat_df = pd.DataFrame(rows)

# === Inspect first few rows ===
print(feat_df)''''''

    #just testing 

if period == 2.1713484:
        print("Value can be drawn.")
    else:
        print("Ignore this.")
    #this is the model so far

'''

#here is the modularized version of the dataFinder
import pandas as pd
import numpy as np
from pathlib import Path

LABEL_MAP = {
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0
}

def load_exoplanet_data(csv_path: str | Path) -> pd.DataFrame:
    """
    filters Kepler exoplanet data to key columns for ML training.
    Keeps: koi_depth, koi_duration, koi_period, koi_prad, koi_disposition.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    #Reads CSV
    df = pd.read_csv(csv_path, comment="#")

    # Filter the important columns
    keep_cols = ["koi_depth", "koi_duration", "koi_period", "koi_prad", "koi_disposition","koi_srad"]
    df = df[keep_cols]

    #map labels
    df["koi_disposition"] = (
        df["koi_disposition"]
        .astype(str)
        .str.upper()
        .map(LABEL_MAP)
    )

    df = df.dropna(subset=["koi_depth", "koi_duration", "koi_period", "koi_prad", "koi_disposition","koi_srad"])

    return df


