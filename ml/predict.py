"""Прогон обученной модели по файлу (например, по закрытому датасету организаторов).

  python -m ml.predict --data data/test.csv --out ml/artifacts/predictions.csv

Если в файле есть колонка label — дополнительно печатает accuracy/precision/recall/F1.
"""
import argparse, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .dataset import read_table, normalize
from .inference import get_model, explain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="ml/artifacts/predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    model = get_model()
    if model is None:
        raise SystemExit("Модель не обучена: сначала python -m ml.train ...")
    df = normalize(read_table(args.data))
    df["proba"] = model.predict_proba(df["full_text"].tolist())[:, 1].round(4)
    df["pred"] = (df["proba"] >= args.threshold).astype(int)
    df["predictors"] = df["full_text"].map(lambda t: "; ".join(x["text_ru"] for x in explain(t, top=3)))
    cols = ["title", "area", "proba", "pred", "predictors"] + (["label"] if "label" in df else [])
    df[cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(df)} строк: {args.out}")
    if "label" in df:
        y, p = df["label"], df["pred"]
        print(json.dumps({"accuracy": round(accuracy_score(y, p), 4),
                          "precision": round(precision_score(y, p, zero_division=0), 4),
                          "recall": round(recall_score(y, p, zero_division=0), 4),
                          "f1": round(f1_score(y, p, zero_division=0), 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
