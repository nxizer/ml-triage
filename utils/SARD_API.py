import requests
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

LANGUAGES = ["c", "csharp", "php", "cplusplus", "java"]
STATES    = ["good", "bad", "mixed"]
LIMIT     = 100
BASE_URL  = "https://samate.nist.gov/SARD/api/test-cases/search"
BASE_DIR  = Path("../dataset")

def format_seconds(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fetch_testcases(language: str, state: str, output_path: Path) -> list:
    if output_path.exists():
        print(f"Found existing JSON for {language}/{state}. Checking if it matches API data")
        
        with open(output_path, encoding="utf-8") as f:
            cached = json.load(f)

        server_total = requests.get(BASE_URL, params={
            "language[]": language,
            "state[]": state,
            "page": 1,
            "limit": 1
        }).json().get("total", 0)

        if len(cached) == server_total:
            print(f"Using existing JSON: {len(cached)} testcases")
            return cached
        print(f"Existing JSON does not match API data ({len(cached)} != {server_total})\n")

    page, cases, page_count = 1, [], None
    while True:
        r = requests.get(BASE_URL, params={
            "language[]": language, "state[]": state, "page": page, "limit": LIMIT
        })
        r.raise_for_status()
        data = r.json()

        if page_count is None:
            page_count = data.get("pageCount", 0)

        batch = data.get("testCases", [])
        if not batch:
            break

        print(f"\r[Page {page}/{page_count}] Fetching {language}/{state}", end="", flush=True)
        cases.extend(batch)
        page += 1
        time.sleep(0.1)

    print(f"\nFetched {len(cases)} testcases for {language}/{state}")
    output_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases

def download_zips(cases: list, download_dir: Path):
    download_dir.mkdir(parents=True, exist_ok=True)
    total = len(cases)
    downloaded = 0

    for idx, case in enumerate(cases, 1):
        url = case.get("download")
        if not url:
            continue

        name = urlsplit(url).path.split("/")[-1]
        dest = download_dir / name

        if dest.exists():
            print(f"\r[{idx}/{total}] Already downloaded: {name}", end="", flush=True)
            continue

        print(f"\r[{idx}/{total}] Downloading: {name}", end="", flush=True)
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                dest.write_bytes(r.content)
            downloaded += 1
        except Exception as e:
            print(f"\n[{idx}/{total}] Could not download {name}, Error: {e}")

    print(f"\nDownload complete for {download_dir.name}. New files downloaded: {downloaded}")

if __name__ == "__main__":
    start_time = time.perf_counter()

    for lang in LANGUAGES:
        for state in STATES:
            output_json = BASE_DIR / f"{lang}_sard_{state}.json"
            download_dir = BASE_DIR / f"{lang}_sard_{state}"

            print(f"\n---- Processing {lang.upper()} | {state.upper()} ----")
            try:
                cases = fetch_testcases(lang, state, output_json)
                download_zips(cases, download_dir)
            except Exception as e:
                print(f"\nError processing {lang}/{state}: {e}")

    elapsed = time.perf_counter() - start_time
    print(f"\nElapsed: {format_seconds(elapsed)}")
