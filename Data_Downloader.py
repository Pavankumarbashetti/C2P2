import os
import pandas as pd

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# CUAD master clauses file
url = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/master_clauses.csv"
local_path = os.path.join(DATA_DIR, "master_clauses.csv")

if not os.path.exists(local_path):
    print("Downloading CUAD master_clauses.csv ...")
    df = pd.read_csv(url)
    df.to_csv(local_path, index=False)
    print(f"Saved to {local_path}")
else:
    print("File already exists:", local_path)
