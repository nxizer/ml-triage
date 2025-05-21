import argparse
import csv
import json
import joblib
import logging
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils import shuffle

CWE_FILTER = (79, 89)
RANDOM_STATE = 42
MODEL_FILE = "catboost_php.joblib"
DEV_PRED_FILE = "php_dev_predictions.csv"
CX_PRED_FILE = "php_cx_predictions.csv"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("")
    p.add_argument("--train-csv", type=Path, required=True, help="Dataset with columns id,label,source,trace,cwe")
    p.add_argument("--cx-csv", type=Path, help="Optional. For inference on Checkmarx dump")
    p.add_argument("--out-dir", type=Path, default=Path("./output"))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

def safe_parse_trace(trace_str: str | None) -> str:
    if not trace_str or not isinstance(trace_str, str):
        return ""
    if trace_str.startswith('[{"') and '""' in trace_str:
        trace_str = trace_str.replace('""', '"')
    try:
        data = json.loads(trace_str)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(trace_str)
        except Exception:
            logging.debug("Failed to parse trace snippet: %s", trace_str[:120])
            return ""
    if isinstance(data, list):
        return " ".join(step.get("sourceCode", "") for step in data if isinstance(step, dict))
    return ""

def load_dataset(path: Path) -> pd.DataFrame:
    logging.info("Loading %s", path)
    df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL)
    df["cwe"] = df["cwe"].astype(int)
    df = df[df["cwe"].isin(CWE_FILTER)].reset_index(drop=True)
    logging.info("Dataset size after CWE filter: %d", len(df))
    return df

def balance_cwes(df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for cwe in CWE_FILTER:
        sub = df[df["cwe"] == cwe]
        if sub["label"].nunique() < 2:
            logging.warning("Skipping CWE %s – lacks both labels", cwe)
            continue
        n = sub["label"].value_counts().min()
        for lbl in (0, 1):
            frames.append(sub[sub["label"] == lbl].sample(n, random_state=RANDOM_STATE))
    if not frames:
        raise ValueError("Balancing produced empty frame – check dataset labels")
    return shuffle(pd.concat(frames, ignore_index=True), random_state=RANDOM_STATE)

def build_vectorizer() -> FeatureUnion:
    word_vect = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[\$\w\->]+",
        ngram_range=(1, 2),
        max_features=20_000,
    )
    char_vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=2,
        max_features=30_000,
    )
    return FeatureUnion([("word", word_vect), ("char", char_vect)])

def build_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.01,
        depth=None,
        random_seed=RANDOM_STATE,
        eval_metric="Accuracy",
        early_stopping_rounds=200,
        loss_function="Logloss",
        task_type="CPU",
        verbose=200,
        thread_count=-1
    )

def build_pipeline() -> Pipeline:
    return Pipeline([("vect", build_vectorizer()), ("clf", build_catboost())])

def train(df: pd.DataFrame, model_file: str, dev_pred_file: str, out_dir: Path) -> Pipeline:
    df["trace_text"] = df["trace"].apply(safe_parse_trace)
    df["text"] = (df["source"].fillna("") + " " + df["trace_text"]).str.strip()

    X, y = df["text"], df["label"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, dev_idx = next(splitter.split(X, y))

    pipe = build_pipeline()
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

    y_dev_pred = pipe.predict(X.iloc[dev_idx])
    y_dev_prob = pipe.predict_proba(X.iloc[dev_idx])[:, 1]

    logging.info("Dev accuracy: %.4f", accuracy_score(y.iloc[dev_idx], y_dev_pred))
    logging.info("\n%s", classification_report(y.iloc[dev_idx], y_dev_pred))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / model_file)
    logging.info("Model saved to: %s", out_dir / model_file)

    pd.DataFrame({
        "id": df.iloc[dev_idx]["id"].values if "id" in df.columns else dev_idx,
        "true_label": y.iloc[dev_idx].values,
        "predicted_label": y_dev_pred,
        "probability": y_dev_prob,
    }).to_csv(out_dir / dev_pred_file, index=False)
    logging.info("Dev predictions saved to: %s", out_dir / dev_pred_file)

    return pipe

def infer_cx(pipe: Pipeline, cx_path: Path, cx_pred_file: str, out_dir: Path) -> None:
    df = pd.read_csv(cx_path, engine="python", quoting=csv.QUOTE_MINIMAL)
    df["trace_text"] = df["trace"].apply(safe_parse_trace)
    df["text"] = (df["source"].fillna("") + " " + df["trace_text"]).str.strip()

    preds = pipe.predict(df["text"])
    probs = pipe.predict_proba(df["text"] )[:, 1]

    pd.DataFrame({
        "id": df["id"],
        "predicted_label": preds.astype(int),
        "probability": probs,
    }).to_csv(out_dir / cx_pred_file, index=False)
    logging.info("CX predictions saved to: %s", out_dir / cx_pred_file)

def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    df = load_dataset(args.train_csv)
    df_bal = balance_cwes(df)

    model = train(df_bal, MODEL_FILE, DEV_PRED_FILE, out_dir=args.out_dir)

    if args.cx_csv:
        infer_cx(model, args.cx_csv, CX_PRED_FILE, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
