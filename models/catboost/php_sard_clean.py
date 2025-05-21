import csv
import logging
from pathlib import Path
import pandas as pd
from php_sard import (
    parse_args,
    configure_logging,
    balance_cwes,
    train,
    infer_cx
)

CWE_FILTER = (79, 89)
RANDOM_STATE = 42
MODEL_FILE = "catboost_php_clean.joblib"
DEV_PRED_FILE = "php_dev_predictions_clean.csv"
CX_PRED_FILE = "php_cx_predictions_clean.csv"
CLEANED_FILE = "train_cleaned.csv"

def deduplicate_by_source(df: pd.DataFrame) -> pd.DataFrame:
    norm = df["source"].fillna("").str.replace(r"\s+", "", regex=True)
    before = len(df)
    df = (
        df.assign(_norm=norm)
          .drop_duplicates("_norm", keep="first")
          .drop(columns="_norm")
          .reset_index(drop=True)
    )
    removed = before - len(df)
    if removed:
        logging.info("Deduplicated %d rows (whitespace-insensitive)", removed)
    return df

def load_dataset(path: Path) -> pd.DataFrame:
    logging.info("Loading %s", path)
    df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL)
    df["cwe"] = df["cwe"].astype(int)
    df = df[df["cwe"].isin(CWE_FILTER)].reset_index(drop=True)
    df = deduplicate_by_source(df)
    logging.info("Dataset size after filter and dedup: %d", len(df))
    return df

def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    df_clean = load_dataset(args.train_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.out_dir / CLEANED_FILE, index=False)
    logging.info("Clean dataset saved to: %s", args.out_dir / CLEANED_FILE)

    df_bal = balance_cwes(df_clean)

    model = train(df_bal, MODEL_FILE, DEV_PRED_FILE, out_dir=args.out_dir)

    if args.cx_csv:
        infer_cx(model, args.cx_csv, CX_PRED_FILE, out_dir=args.out_dir)

if __name__ == "__main__":
    main()