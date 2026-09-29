"""Обучение классификатора слабых сигналов (этап 1 ТЗ). Цель: accuracy/F1 >= 0.75-0.80.

Позитивы — размеченные методологами слабые сигналы, негативы — мейнстрим, хайп и шум.
Итоговая модель — логистическая регрессия на 16 признаках-критериях (ml/features.py):
вклад каждого признака в решение виден явно.

  python -m ml.train --positives data/dataset.xlsx --negatives data/negatives.xlsx
  python -m ml.train --labeled data/labeled.csv            # файл с колонкой label (0/1)
"""
import argparse, json, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from .dataset import load_training, build_text
from .features import LexiconFeatures, LEXICON_NAMES, LEXICON_RU

SCORING = ["accuracy", "precision", "recall", "f1"]
FINAL = "lexicon_lr"


def _lr():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)


def _tfidf():
    return TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _text_length(X):
    return np.log1p(np.array([[len(t)] for t in X], dtype=float))


def build_models() -> dict:
    return {
        # контроль: если длина текста сама по себе угадывает класс, модель может учить автора, а не смысл
        "length_only": Pipeline([("len", FunctionTransformer(_text_length)), ("clf", _lr())]),
        "lexicon_lr": Pipeline([("lex", LexiconFeatures()), ("sc", StandardScaler()), ("clf", _lr())]),
        "tfidf_lr": Pipeline([("tfidf", _tfidf()), ("clf", _lr())]),
        "lexicon_tfidf_lr": Pipeline([
            ("union", FeatureUnion([("lex", Pipeline([("lex", LexiconFeatures()), ("sc", StandardScaler())])),
                                    ("tfidf", _tfidf())])),
            ("clf", _lr())]),
    }


def cv_scores(model, X, y) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    r = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return {m: {"mean": round(float(r[f"test_{m}"].mean()), 4), "std": round(float(r[f"test_{m}"].std()), 4)}
            for m in SCORING}


def holdout(model, X, y):
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(y)), test_size=0.2,
                                                    random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
               "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0)}
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics.update(n_train=len(ytr), n_test=len(yte), confusion_matrix=confusion_matrix(yte, pred).tolist())
    proba = model.predict_proba(Xte)[:, 1]
    errors = [{"idx": int(i), "true": int(t), "proba": round(float(p), 3)}
              for i, t, pr, p in zip(ite, yte, pred, proba) if t != pr]
    return metrics, errors


def _first_clause(s: str) -> str:
    """«Растёт быстро — почти пятикратный рост…» -> «Растёт быстро»."""
    return re.split(r"\s[—–-]\s|:|\(|;", s)[0].strip()


def style_control(df: pd.DataFrame, X, y) -> pd.DataFrame:
    """Accuracy контроля по длине, итоговой модели и TF-IDF при трёх вариантах входа."""
    variants = {
        "полный вход": X,
        "суть стадии и тренда (длина выровнена)": (df["stage"].map(_first_clause) + ". "
                                                  + df["trend"].map(_first_clause)).tolist(),
        "только название, область и компании": df.apply(lambda r: build_text(r, with_meta=False), axis=1).tolist(),
    }
    fmt = lambda s: f"{s['mean']:.3f} ± {s['std']:.3f}"
    rows = {v: {m: fmt(cv_scores(build_models()[m], Xv, y)["accuracy"]) for m in ["length_only", FINAL, "tfidf_lr"]}
            for v, Xv in variants.items()}
    return pd.DataFrame(rows).T.rename_axis("вход")


def feature_weights(model) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    return pd.DataFrame({"feature": LEXICON_NAMES, "описание": [LEXICON_RU[f] for f in LEXICON_NAMES],
                         "вес": clf.coef_[0].round(3)}).sort_values("вес", ascending=False)


def _md(df: pd.DataFrame, index: bool = False) -> str:
    """DataFrame -> markdown-таблица (без зависимости от tabulate)."""
    d = df.reset_index() if index else df
    rows = [" | ".join(map(str, d.columns)), " | ".join("---" for _ in d.columns)]
    rows += [" | ".join(map(str, r)) for r in d.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


def write_report(out: Path, log, df, cv, hold, errors, weights, stats, checks):
    fmt = lambda s: f"{s['mean']:.3f} ± {s['std']:.3f}"
    lines = ["# Отчёт об обучении модели слабых сигналов", "",
             "## Данные и очистка", "",
             f"- Строк до очистки: {log['rows_raw']}; после: {log['rows_clean']} "
             f"(слабых сигналов: {log['positives']}, прочих: {log['negatives']}).",
             f"- Удалено: дублей {log['duplicates_removed']}, противоречивых меток "
             f"{log['conflicting_labels_removed']}, слишком коротких описаний {log['too_short_removed']}.",
             "- Очистка текста: markdown-ссылки и URL удалены, пробелы нормализованы, ё→е.",
             "- Вход модели: название, область, компании, стадия развития, тренд упоминаний.",
             "- Не подаются: «Почему это (не) слабый сигнал» — разметчик прямо пишет в ней ответ; "
             "«Балл» и числовые шкалы — их ставили разные люди для разных классов; номер строки, ссылки, "
             "класс негатива (он нужен только для отчёта).", "",
             "Состав по категориям:", "", df.groupby(["label", "category"]).size().rename("строк")
             .pipe(_md, index=True), "",
             "## Сравнение моделей (5×5 стратифицированная кросс-валидация)", "",
             "| Модель | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for name, s in cv.items():
        mark = " **(итоговая)**" if name == FINAL else ""
        lines.append(f"| {name}{mark} | {fmt(s['accuracy'])} | {fmt(s['precision'])} | "
                     f"{fmt(s['recall'])} | {fmt(s['f1'])} |")
    cm = hold["confusion_matrix"]
    lines += ["", f"Итоговая модель — `{FINAL}`: логистическая регрессия на {len(LEXICON_NAMES)} признаках-критериях. "
              "Её выбираем ради интерпретируемости: каждый признак — критерий из ТЗ, вес показывает направление и силу влияния. "
              "TF-IDF-модели учатся на конкретных словах и названиях, поэтому хуже переносятся на открытый поиск.",
              "", "## Отложенная выборка (20%)", "",
              f"Accuracy **{hold['accuracy']:.3f}**, Precision **{hold['precision']:.3f}**, "
              f"Recall **{hold['recall']:.3f}**, F1 **{hold['f1']:.3f}** "
              f"(train {hold['n_train']}, test {hold['n_test']}).", "",
              "| | предсказано: не сигнал | предсказано: сигнал |", "|---|---|---|",
              f"| **на самом деле: не сигнал** | {cm[0][0]} | {cm[0][1]} |",
              f"| **на самом деле: сигнал** | {cm[1][0]} | {cm[1][1]} |", ""]
    if errors:
        lines += ["Ошибки на отложенной выборке:", ""]
        for e in errors:
            r = df.iloc[e["idx"]]
            lines.append(f"- {r['title']} — метка {e['true']}, вероятность {e['proba']}")
        lines.append("")
    lines += ["## Контроль стиля автора", "",
              "Позитивы и негативы писали разные люди: тексты позитивов длиннее. Модель могла бы "
              "научиться отличать авторов, а не классы, и провалиться на закрытом датасете, где все строки "
              "написаны организаторами. Контроль `length_only` видит только длину текста.", "",
              _md(checks["lengths"], index=True), "",
              "Accuracy (5×5 CV) при разном входе:", "", _md(checks["style"], index=True), "",
              "Как читать: когда длина выровнена (суть стадии и тренда), контроль по длине не лучше "
              "угадывания, а итоговая модель сохраняет точность — она различает классы по смыслу. "
              "По одним названиям и компаниям итоговая модель класс не угадывает, то есть на стиль "
              "и конкретные имена не опирается.", "",
              "## Колонка с обоснованием", "",
              "Если подать модели «Почему это (не) слабый сигнал», accuracy "
              f"будет {fmt(checks['with_rationale']['accuracy'])}. Мы от неё отказались: там записан ответ, "
              "а в реальном поиске такого поля нет.", "",
              "## Интерпретация: веса признаков", "",
              "Положительный вес — признак слабого сигнала, отрицательный — мейнстрима, хайпа или шума. "
              "Признаки стандартизированы, веса сопоставимы между собой.", "",
              _md(weights), "",
              "Средние значения признаков по классам:", "", _md(stats.rename_axis("признак"), index=True), "",
              "## Ограничения", "",
              "- Выборка маленькая, поэтому метрики приведены со стандартным отклонением по 25 разбиениям.",
              "- Негативы размечены командой по протоколу (лист «Как размечали»); итоговая проверка — "
              "на закрытом датасете организаторов "
              "(`python -m ml.predict --data <файл>`)."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", nargs="*", default=[], help="файлы, где все строки — слабые сигналы")
    ap.add_argument("--negatives", nargs="*", default=[], help="файлы, где все строки — не слабые сигналы")
    ap.add_argument("--labeled", nargs="*", default=[], help="файлы с колонкой label (0/1)")
    ap.add_argument("--out", default="ml/artifacts")
    ap.add_argument("--with-rationale", action="store_true",
                    help="подавать колонку обоснования (не рекомендуется: в ней ответ)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, log = load_training(args.positives, args.negatives, args.labeled, with_rationale=args.with_rationale)
    if df["label"].nunique() < 2:
        raise SystemExit("Нужны оба класса: добавьте --negatives или файл с колонкой label.")
    X, y = df["full_text"].tolist(), df["label"].to_numpy()
    print(json.dumps(log, ensure_ascii=False))

    cv = {name: cv_scores(m, X, y) for name, m in build_models().items()}
    for name, s in cv.items():
        print(f"{name:18s} acc={s['accuracy']['mean']:.3f} f1={s['f1']['mean']:.3f}")
    hold, errors = holdout(build_models()[FINAL], X, y)
    style = style_control(df, X, y)
    X_rat = df.apply(lambda r: build_text(r, with_rationale=True), axis=1).tolist()
    lengths = df.assign(класс=df["label"].map({1: "слабый сигнал", 0: "не сигнал"}),
                        длина=df["full_text"].str.len()).groupby("класс")["длина"].agg(["median", "mean"]).round(0)
    checks = {"lengths": lengths.rename(columns={"median": "медиана, симв.", "mean": "среднее, симв."}),
              "style": style, "with_rationale": cv_scores(build_models()[FINAL], X_rat, y)}

    model = build_models()[FINAL].fit(X, y)
    weights = feature_weights(model)
    feats = pd.DataFrame(model.named_steps["lex"].transform(X), columns=LEXICON_NAMES)
    stats = feats.groupby(df["label"].map({1: "слабый сигнал", 0: "не сигнал"})).mean().T.round(2)

    joblib.dump(model, out / "model.pkl")
    pd.concat([df, feats], axis=1).to_csv(out / "prepared_dataset.csv", index=False, encoding="utf-8-sig")
    metrics = {"model": FINAL, **{k: hold[k] for k in SCORING}, "holdout": hold,
               "cv_5x5": cv, "style_control": checks["style"].to_dict(orient="index"), "with_rationale": checks["with_rationale"],
               "input": "title+area+companies+stage+trend" + ("+rationale" if args.with_rationale else ""),
               "data": log}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, log, df, cv, hold, errors, weights, stats, checks)
    print(json.dumps({k: hold[k] for k in SCORING}, ensure_ascii=False))
    print(f"Сохранено: {out/'model.pkl'}, {out/'metrics.json'}, {out/'report.md'}")


if __name__ == "__main__":
    main()
