import os
import pandas as pd

DATA_DIR = "/home/nxizer/workdir/ml-triage/dataset/"

splits = ["train", "test", "dev"]

combined = {split: [] for split in splits}

for filename in os.listdir(DATA_DIR):
    for split in splits:
        if filename.endswith(f"{split}.csv") and filename.startswith("splitdata_"):
            file_path = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(file_path)

            project_name = filename.replace("splitdata_", "").replace(f"_{split}.csv", "")
            df["project"] = project_name
            df["split"] = split

            combined[split].append(df)

for split in splits:
    if combined[split]:
        df_concat = pd.concat(combined[split], ignore_index=True)
        output_path = os.path.join(DATA_DIR, f"{split}.csv")
        df_concat.to_csv(output_path, index=False)
        print(f"Saved {split}.csv with {len(df_concat)} rows")
    else:
        print(f"No data found for {split}")
