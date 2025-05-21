import zipfile
import json
import csv
import time
import shutil
from pathlib import Path
import pandas as pd
from SARD_API import format_seconds

LANGUAGES = ["csharp", "c", "php", "cplusplus", "java"]
STATES    = ["good", "bad", "mixed"]

BASE_INPUT_DIR  = Path("../dataset/sard_testcases")
BASE_OUTPUT_DIR = Path("../dataset")
JSON_DIR        = BASE_OUTPUT_DIR / "sard_prepared"
JSON_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_NAMES = {
    "csharp": "C#",
    "c": "C",
    "php": "PHP",
    "cplusplus": "C++",
    "java": "Java"
}


def extract_source_line(file_path: Path, line_num: int) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[line_num - 1].rstrip("\n") if 1 <= line_num <= len(lines) else "// ERROR: Line out of range"
    except Exception:
        return "// ERROR: Cannot read line"

def process_sarif(sarif_data: dict, extracted_dir: Path) -> dict | None:
    result = {"path": [], "category": {"cwes": []}}
    for taxonomy in sarif_data.get("runs", [])[0].get("taxonomies", []):
        if taxonomy.get("name") == "CWE":
            for taxon in taxonomy.get("taxa", []):
                result["category"] = {
                    "category": taxon.get("name", ""),
                    "cwes": [{
                        "id": str(taxon.get("id", "")),
                        "link": f"https://cwe.mitre.org/data/definitions/{taxon.get('id')}.html"
                    }],
                }
    runs = sarif_data.get("runs", [])
    if not runs:
        return None
    sarif_results = runs[0].get("results", [])
    if not sarif_results:
        return None
    first_result = sarif_results[0]
    locations = []
    sequence = 1
    if "codeFlows" in first_result:
        for code_flow in first_result["codeFlows"]:
            for thread_flow in code_flow.get("threadFlows", []):
                for loc in thread_flow.get("locations", []):
                    physical = loc.get("location", {}).get("physicalLocation", {})
                    artifact_path = physical.get("artifactLocation", {}).get("uri")
                    line = physical.get("region", {}).get("startLine")
                    if not artifact_path or line is None:
                        continue
                    src_file = extracted_dir / artifact_path
                    src_line = extract_source_line(src_file, line)
                    locations.append({"fileName": artifact_path, "line": line, "sourceCode": src_line, "sequence": sequence})
                    sequence += 1
    else:
        for loc in first_result.get("locations", []):
            physical = loc.get("physicalLocation", {})
            artifact_path = physical.get("artifactLocation", {}).get("uri")
            line = physical.get("region", {}).get("startLine")
            if not artifact_path or line is None:
                continue
            src_file = extracted_dir / artifact_path
            src_line = extract_source_line(src_file, line)
            locations.append({"fileName": artifact_path, "line": line, "sourceCode": src_line, "sequence": sequence})
            sequence += 1
    if not locations:
        return None
    result["path"] = locations
    return result

def process_archives_for_state(language: str, state: str) -> list:
    label = 0 if state == "good" else 1
    input_dir = BASE_INPUT_DIR / f"{language}_sard_{state}"
    all_state_records = []
    success_count = 0
    processed = 0
    total_archives = len(list(input_dir.glob("*.zip")))
    for zip_path in input_dir.glob("*.zip"):
        extract_path = input_dir / f"_extracted_{zip_path.stem}"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            sarif_file = next(extract_path.rglob("*.sarif"), None)
            if sarif_file is None:
                raise ValueError("SARIF файл не найден")
                processed += 1
            with open(sarif_file, "r", encoding="utf-8") as f:
                sarif_data = json.load(f)
                processed += 1
            parsed = process_sarif(sarif_data, extract_path)
            if parsed is None:
                continue
                processed += 1
            parsed.update({"label": label, "language": language, "zip": zip_path.name})
            all_state_records.append(parsed)
            success_count += 1
        except Exception as e:
            print(f"ERROR: {language} | {state}: into {zip_path.name}: {e}")
        finally:
            shutil.rmtree(extract_path, ignore_errors=True)
        print(f"[{processed}/{total_archives}] Processing {state.upper()} testcases", end='\r', flush=True)
    print(f"Processed all {state.upper()} testcases. Succeeded {success_count}/{total_archives}")
    return all_state_records

def save_json(records: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def json_to_csv(records: list, csv_path: str) -> None:
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "source", "trace", "cwe"])
        writer.writeheader()
        for idx, item in enumerate(records, 1):
            trace = item.get("path", [])
            source_code = trace[-1]["sourceCode"] if trace else ""
            cwe_list = item.get("category", {}).get("cwes", [])
            cwe_id = cwe_list[0]["id"] if cwe_list else ""
            writer.writerow({
                "id": idx,
                "label": item.get("label", -1),
                "source": source_code,
                "trace": json.dumps(trace, ensure_ascii=False),
                "cwe": cwe_id
            })

def remove_duplicates(input_csv: str):
    df = pd.read_csv(input_csv)
    if "id" not in df.columns:
        print("WARNING: could not find 'id' column in CSV. No duplicates were removed.")
        return
    df_dedup = df.drop_duplicates(subset=[col for col in df.columns if col != "id"])
    removed = len(df) - len(df_dedup)
    df_dedup.to_csv(input_csv, index=False)
    print(f"Duplicates removed: {removed}. CSV saved to: {input_csv}")

if __name__ == "__main__":
    total_start = time.perf_counter()
    for lang in LANGUAGES:
        pretty_lang = LANGUAGE_NAMES.get(lang, lang.upper())
        print(f"\n---- {pretty_lang} ----")
        combined_records = []
        for state in STATES:
            combined_records.extend(process_archives_for_state(lang, state))
        combined_json_path = JSON_DIR / f"{lang}_combined.json"
        csv_dedup_path = BASE_OUTPUT_DIR / f"{lang}_sard.csv"
        save_json(combined_records, combined_json_path)
        print(f"\n{pretty_lang} JSON saved to: {combined_json_path}")
        json_to_csv(combined_records, str(csv_dedup_path))
        remove_duplicates(str(csv_dedup_path))
        print(f"Finished processing {pretty_lang}. Total lines in CSV count: {len(combined_records)}")
    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll languages processed. Elapsed: {format_seconds(total_elapsed)}")
