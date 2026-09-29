"""Загрузка и очистка датасета (этап 1 ТЗ: «Предобработка: очистка шума»).

Приводит xlsx/csv/json с произвольной шапкой к единой таблице:
title, area, companies, rationale, stage, trend, label, full_text.
Колонки сопоставляются по смыслу (подстрока в названии), поэтому подходят
и наш xlsx, и закрытый датасет организаторов в CSV/JSON.
"""
import re
from pathlib import Path
import pandas as pd

FIELDS = {  # поле -> подстроки в названии колонки
    "title": ["технолог", "название", "сигнал", "title", "name", "technology"],
    "area": ["область", "сфера", "отрасль", "area", "domain", "industry"],
    "companies": ["компани", "игроки", "companies", "players"],
    "rationale": ["почему", "обоснован", "описан", "description", "abstract", "rationale", "summary", "text"],
    "stage": ["стадия", "зрелост", "stage", "trl", "maturity"],
    "trend": ["тренд", "динамик", "trend"],
}
LABEL_NAMES = {"label", "is_weak", "weak_signal", "is_weak_signal", "target", "y", "метка"}
# Не используются как признаки: номер строки, «Балл» (у позитивов и негативов его ставили
# разные люди), ссылки, класс негатива (мейнстрим/хайп/шум) — он идёт только в отчёт.
IGNORE = ["№", "балл", "score", "источник", "source", "url", "ссылк", "категори", "category", "класс", "class"]
CATEGORY_KEYS = ["класс", "категори", "category", "class"]
POSITIVE_VALUES = {"1", "1.0", "true", "yes", "да", "weak", "слабый сигнал"}


def _find_header(raw: pd.DataFrame) -> int:
    """Номер строки с шапкой: в xlsx над ней может быть заголовок таблицы."""
    for i in range(min(10, len(raw))):
        cells = [str(c).lower() for c in raw.iloc[i] if pd.notna(c)]
        if len(cells) >= 3 and any(k in c for c in cells for k in FIELDS["title"]):
            return i
    return 0


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None)
        h = _find_header(raw)
        df = raw.iloc[h + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[h]]
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and c.lower() not in ("nan", "none")]]
    return df.dropna(how="all")


def clean_text(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # markdown-ссылки -> текст
    s = re.sub(r"https?://\S+", " ", s)               # голые URL
    return re.sub(r"\s+", " ", s).strip()


def _is_numeric(col: pd.Series) -> bool:
    return pd.to_numeric(col, errors="coerce").notna().mean() > 0.8


def _map_columns(df: pd.DataFrame) -> dict:
    """Поле схемы -> колонка файла. Числовые шкалы («Стадия (0–5)») текст не подменяют."""
    mapping, used = {}, set()
    for field, keys in FIELDS.items():
        for c in df.columns:
            lc = c.lower()
            if c in used or any(x in lc for x in IGNORE) or lc in LABEL_NAMES or _is_numeric(df[c]):
                continue
            if any(k in lc for k in keys):
                mapping[field] = c
                used.add(c)
                break
    return mapping


def build_text(row, with_meta: bool = True, with_rationale: bool = False) -> str:
    """Текст наблюдения для модели.
    with_rationale=False (по умолчанию): колонки «Почему это (не) слабый сигнал» не подаются —
    в них разметчик прямо пишет ответ. with_meta=False — ещё и без стадии/тренда (стресс-тест)."""
    parts = [row["title"] + ".", f"Область: {row['area']}." if row["area"] else "",
             f"Компании: {row['companies']}." if row["companies"] else "",
             row["rationale"] if with_rationale else ""]
    if with_meta:
        parts += [f"Стадия: {row['stage']}." if row["stage"] else "",
                  f"Тренд: {row['trend']}." if row["trend"] else ""]
    return " ".join(p for p in parts if p)


def normalize(df: pd.DataFrame, label: int | None = None) -> pd.DataFrame:
    """Сырая таблица -> единая схема. label — метка для файла без колонки меток."""
    mapping = _map_columns(df)
    out = pd.DataFrame({f: df[mapping[f]].map(clean_text) if f in mapping else "" for f in FIELDS},
                       index=df.index)
    if "rationale" not in mapping:  # нет явного описания — склеиваем прочие текстовые колонки
        rest = [c for c in df.columns if c not in mapping.values() and not _is_numeric(df[c])
                and not any(x in c.lower() for x in IGNORE) and c.lower() not in LABEL_NAMES]
        out["rationale"] = df[rest].astype(str).map(clean_text).agg(" ".join, axis=1) if rest else ""
    label_col = next((c for c in df.columns if c.lower() in LABEL_NAMES), None)
    if label_col is not None:
        out["label"] = df[label_col].astype(str).str.strip().str.lower().isin(POSITIVE_VALUES).astype(int)
    elif label is not None:
        out["label"] = int(label)
    cat_col = next((c for c in df.columns if any(k in c.lower() for k in CATEGORY_KEYS)), None)
    out["category"] = df[cat_col].map(clean_text) if cat_col else ""
    out = out[out["title"] != ""]
    out["full_text"] = out.apply(build_text, axis=1)
    return out.reset_index(drop=True)


def load_training(positives=(), negatives=(), labeled=(), with_rationale=False) -> tuple[pd.DataFrame, dict]:
    """Собирает обучающую выборку и журнал очистки."""
    frames = []
    for paths, lab in ((positives, 1), (negatives, 0), (labeled, None)):
        for p in paths:
            d = normalize(read_table(p), label=lab)
            d["file"] = Path(p).name
            if lab == 1:
                d["category"] = d["category"].replace("", "слабый сигнал")
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    log = {"rows_raw": len(df)}
    df = df[df["label"].notna()]
    key = df["title"].str.lower().str.replace("ё", "е").str.replace(r"[^\w]+", " ", regex=True).str.strip()
    conflicts = key[key.duplicated(keep=False)].groupby(key).apply(
        lambda s: df.loc[s.index, "label"].nunique() > 1)
    bad = set(conflicts[conflicts].index)
    log["conflicting_labels_removed"] = int(key.isin(bad).sum())
    df = df[~key.isin(bad)]
    key = key[df.index]
    log["duplicates_removed"] = int(key.duplicated().sum())
    df = df[~key.duplicated()]
    short = df["full_text"].str.len() < 40
    log["too_short_removed"] = int(short.sum())
    df = df[~short].reset_index(drop=True)
    df["full_text"] = df.apply(build_text, axis=1, with_rationale=with_rationale)
    df["label"] = df["label"].astype(int)
    log.update(rows_clean=len(df), positives=int(df["label"].sum()),
               negatives=int((df["label"] == 0).sum()))
    return df, log
