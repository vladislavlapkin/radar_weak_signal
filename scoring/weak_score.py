"""Правила WSS (Weak Signal Score) по кластеру источников + понятное объяснение каждого слагаемого."""
import math
from datetime import date, timedelta

WEIGHTS = {"novelty": 0.30, "recency": 0.25, "diversity": 0.15, "science": 0.15, "maturity": 0.35, "hype": 0.25}
POSITIVE_MAX = WEIGHTS["novelty"] + WEIGHTS["recency"] + WEIGHTS["diversity"] + WEIGHTS["science"]
RECENT_DAYS = 548  # 18 месяцев


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _date(s: str | None) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def rule_features(docs: list[dict], maturity: float, hype: float) -> dict:
    dated = [d for d in map(lambda x: _date(x.get("published_at")), docs) if d]
    cutoff = date.today() - timedelta(days=RECENT_DAYS)
    high = sum(1 for d in docs if d["trust_level"] == "high")
    return {"n_docs": len(docs), "n_dated": len(dated), "n_recent": sum(1 for d in dated if d >= cutoff),
            "n_types": len({d["source_type"] for d in docs}), "n_high": high,
            "science_share": high / max(1, len(docs)), "maturity": _clip(maturity), "hype": _clip(hype)}


def compute_weak_score(f: dict) -> tuple[float, list[dict]]:
    n = f["n_docs"]
    novelty = _clip(1 - math.log1p(n) / math.log1p(30))
    recency = f["n_recent"] / f["n_dated"] if f["n_dated"] else 0.5
    diversity = _clip((f["n_types"] / 4 + min(n, 8) / 8) / 2)
    science = f["science_share"]
    pos = (WEIGHTS["novelty"] * novelty + WEIGHTS["recency"] * recency
           + WEIGHTS["diversity"] * diversity + WEIGHTS["science"] * science) / POSITIVE_MAX
    score = _clip(pos - WEIGHTS["maturity"] * f["maturity"] - WEIGHTS["hype"] * f["hype"])
    recency_txt = (f"{f['n_recent']} из {f['n_dated']} публикаций с датой — не старше 18 месяцев"
                   if f["n_dated"] else "у источников нет дат — оценка нейтральная")
    parts = [
        {"label": "Новизна", "value": round(novelty, 2), "effect": "+",
         "detail": f"{n} {_docs_word(n)} по теме — чем меньше упоминаний, тем раньше стадия"},
        {"label": "Свежесть", "value": round(recency, 2), "effect": "+", "detail": recency_txt},
        {"label": "Разнообразие источников", "value": round(diversity, 2), "effect": "+",
         "detail": f"{f['n_types']} {_types_word(f['n_types'])} источников — защита от вброса из одного канала"},
        {"label": "Научная база", "value": round(science, 2), "effect": "+",
         "detail": f"{f['n_high']} из {n} источников — высокого доверия: статьи, патенты, гос- и edu-сайты"},
    ]
    if f["maturity"]:
        parts.append({"label": "Штраф за зрелость", "value": round(f["maturity"], 2), "effect": "−",
                      "detail": "в источниках есть маркеры стандартов, лидеров рынка или массового внедрения"})
    if f["hype"]:
        parts.append({"label": "Штраф за хайп", "value": round(f["hype"], 2), "effect": "−",
                      "detail": "громкие маркетинговые формулировки"})
    return round(score, 4), parts


def _docs_word(n: int) -> str:
    return "документ" if n % 10 == 1 and n % 100 != 11 else (
        "документа" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "документов")


def _types_word(n: int) -> str:
    return "тип" if n % 10 == 1 and n % 100 != 11 else ("типа" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "типов")
