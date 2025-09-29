import json, pandas as pd

# Load features list from training metadata
with open("./models/run1/run_summary.json") as f:
    meta = json.load(f)

used = meta["used_features"]

# Load full dataset, drop unwanted columns
df = pd.read_csv("data.csv").drop(columns=["diagnosis", "id", "Unnamed: 32"], errors="ignore")

# Keep only the features used during training
df[used].head(5).to_csv("sample_input.csv", index=False)

print("sample_input.csv created with correct features!")
