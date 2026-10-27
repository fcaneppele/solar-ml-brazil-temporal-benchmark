"""Download the NASA/POWER daily point data used in the study (optional).

The files in data/raw/ are the exact inputs used in the paper (retrieved on 2026-10-07).
NASA/POWER may reprocess its products, so a new download can differ slightly from
that snapshot. Use --force to overwrite the stored files.

Usage:  python src/00_download_power.py [--force]
"""
from pathlib import Path
import os, sys, time
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

START, END = "20200101", "20251231"
PARAMS = ["ALLSKY_SFC_SW_DWN", "TOA_SW_DWN", "T2M", "T2M_MAX", "T2M_MIN",
          "RH2M", "WS2M", "PS", "PRECTOTCORR", "CLOUD_AMT"]
CITIES = {
    "Manaus": (-3.119, -60.022), "Fortaleza": (-3.732, -38.527), "Recife": (-8.054, -34.881),
    "Brasilia": (-15.794, -47.882), "Cuiaba": (-15.601, -56.097), "Belo Horizonte": (-19.917, -43.935),
    "Rio de Janeiro": (-22.907, -43.173), "Sao Paulo": (-23.550, -46.633),
    "Curitiba": (-25.428, -49.273), "Porto Alegre": (-30.035, -51.218),
}


def fetch_power(lat, lon, retries=4):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    q = {"parameters": ",".join(PARAMS), "community": "RE", "longitude": lon, "latitude": lat,
         "start": START, "end": END, "format": "JSON", "time-standard": "LST"}
    for k in range(retries):
        try:
            r = requests.get(url, params=q, timeout=120)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["properties"]["parameter"])
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            return df.replace(-999.0, np.nan)
        except Exception as e:
            print(f"  attempt {k + 1} failed: {e}")
            time.sleep(5 * (k + 1))
    raise RuntimeError("NASA/POWER download failed")


if __name__ == "__main__":
    force = "--force" in sys.argv
    os.makedirs("data/raw", exist_ok=True)
    for city, (lat, lon) in CITIES.items():
        path = f"data/raw/{city.replace(' ', '_')}.csv"
        if os.path.exists(path) and not force:
            print(f"{city}: kept existing {path}")
            continue
        print(f"{city}: downloading...")
        fetch_power(lat, lon).to_csv(path)
