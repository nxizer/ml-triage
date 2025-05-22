import json
import csv
from pathlib import Path
from make_sard_dataset import json_to_csv

LANGUAGE = "php"  # ["csharp", "c", "php", "cplusplus", "java"]
INPUT_JSON = "../dataset/cx/FP_79.json"
OUTPUT_CSV = f"../dataset/{LANGUAGE}_cx.csv"
FORCE_LABEL = 0  # {FP=0, TP=1}

def load_json_items(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_cwe_id(item: dict) -> str:
    category = item.get("category", {})
    cwes = category.get("cwes", [])
    if isinstance(cwes, list) and cwes:
        return cwes[0].get("id", "")
    return ""

if __name__ == "__main__":
    records = load_json_items(INPUT_JSON)
    for item in records:
        item["label"] = FORCE_LABEL

    json_to_csv(records, OUTPUT_CSV)
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"JSON processed. Total lines in CSV count: {len(records)}")
