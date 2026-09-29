"""Карточка кандидата: скоринг (правила + ML), вердикт фильтра и читаемый отчёт.

Тексты строятся в два режима. С YandexGPT: название, описание, преимущество, кейс и обоснование
пишет модель строго по источникам кластера. Без LLM: экстрактивно — цитаты из самих источников
с пометкой языка оригинала. В карточке всегда указано, каким способом получен текст.
"""
import re
from datetime import date
from ml.features import LEXICON_RU
from ml.inference import explain, get_model, predict_proba
from scoring.maturity_filter import STATUS_RU, markers, maturity_check
from scoring.trust import TRUST_RU, TYPE_RU
from scoring.weak_score import compute_weak_score, rule_features
from llm import router

ML_WEIGHT = 0.4
ADVANTAGE_CUES = re.compile(r"improv|outperform|reduc|faster|speed.?up|efficien|accura|enabl|lower cost|cheaper|"
                            r"robust|scalab|позволя|снижа|повыша|ускор|эконом|сокраща|точн", re.I)
TRUST_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", " ".join((text or "").split()))
    return " ".join(p for p in parts[:n] if p)[:420]


def _ranked(docs: list[dict]) -> list[dict]:
    """Сначала высокое доверие, внутри — свежие."""
    return sorted(docs, key=lambda d: (TRUST_ORDER.get(d["trust_level"], 3), -(int((d.get("published_at") or "0")[:4] or 0))))


def confidence_ru(score: float) -> str:
    return "Высокая" if score > 0.75 else ("Средняя" if score > 0.5 else "Низкая")


def score_candidate(c: dict) -> dict:
    """Быстрая часть без LLM: признаки, скоринг, вердикт. Определяет, попадёт ли кандидат в ТОП."""
    g, text = c["docs"], c["combined_text"]
    mature, hype = markers(text)
    f = rule_features(g, maturity=min(1.0, len(mature) / 2), hype=min(1.0, len(hype) / 2))
    rule_score, rule_parts = compute_weak_score(f)
    has_model = get_model() is not None
    ml_p = predict_proba(text)[0] if has_model else None
    ml_top = explain(text, top=5) if has_model else []
    score = round((1 - ML_WEIGHT) * rule_score + ML_WEIGHT * ml_p, 4) if has_model else rule_score
    high = f["n_high"] / max(1, len(g))
    low_only = f["n_high"] == 0 and all(d["trust_level"] == "low" for d in g)
    status, reason = maturity_check(text, len(g), high, low_only, ml_p, ml_top, titles=[d["title"] for d in g])
    return {**c, "features": f, "rule_score": rule_score, "rule_parts": rule_parts, "ml_proba": ml_p,
            "ml_top": ml_top, "score": score, "status": status, "status_reason": reason}


def _predictors(s: dict) -> list[dict]:
    out = [{"label": x["short"], "effect": "+" if x["contribution"] >= 0 else "−",
            "value": round(abs(x["contribution"]), 2), "detail": LEXICON_RU[x["feature"]], "source": "модель"}
           for x in s["ml_top"]]
    out += [{**p, "source": "правила"} for p in s["rule_parts"]]
    return out


def _key_chips(s: dict) -> list[str]:
    """Три коротких предиктора для строки таблицы: сильнейшие доводы в пользу вердикта."""
    want_positive = s["status"] in ("weak", "low_trust")
    ml = [x["short"] for x in s["ml_top"] if (x["contribution"] > 0) == want_positive]
    rules = {"Свежесть": "Свежие публикации", "Научная база": "Научная база", "Новизна": "Мало упоминаний"}
    extra = [rules[p["label"]] for p in sorted(s["rule_parts"], key=lambda p: -p["value"])
             if p["label"] in rules and p["value"] >= 0.6] if want_positive else []
    chips = []
    for x in ml + extra:
        if x not in chips:
            chips.append(x)
    return chips[:3]


def _source_card(d: dict, summary: str | None, model: str | None) -> dict:
    if summary:
        text, note = summary, f"Генеративное резюме · {model}" + (" · авто-перевод" if d["lang"] != "ru" else "")
    elif d["lang"] == "ru":
        text, note = _sentences(d.get("snippet"), 2), "Выдержка из источника"
    else:
        text, note = _sentences(d.get("snippet"), 2), "Оригинал на английском — перевод появится с YandexGPT"
    return {"title": d["title"], "url": d["url"], "site": d["source_name"], "date": d.get("published_at"),
            "type": d["source_type"], "type_ru": TYPE_RU.get(d["source_type"], d["source_type"]),
            "lang": d["lang"].upper(), "trust": d["trust_level"], "trust_ru": TRUST_RU.get(d["trust_level"], ""),
            "summary": text or "Аннотация недоступна", "summary_note": note}


def _offline_texts(s: dict, docs: list[dict]) -> dict:
    f, lead = s["features"], docs[0]
    years = sorted({d["published_at"][:4] for d in docs if d.get("published_at")})
    period = f"{years[0]}–{years[-1]}" if len(years) > 1 else (years[0] if years else "без дат")
    kinds = ", ".join(sorted({TYPE_RU.get(d["source_type"], d["source_type"]).lower() for d in docs}))
    description = (f"Тема «{s['tech_name']}» найдена в {f['n_docs']} {_src_word(f['n_docs'])} ({kinds}), "
                   f"период публикаций: {period}.")
    adv = next(((sent, d) for d in docs for sent in re.split(r"(?<=[.!?])\s+", d.get("snippet") or "")
                if ADVANTAGE_CUES.search(sent) and 40 < len(sent) < 400), None)
    return {
        "description": description,
        "quote": {"text": _sentences(lead.get("snippet"), 2), "lang": lead["lang"].upper(), "site": lead["source_name"]}
        if lead.get("snippet") else None,
        "advantage": adv[0].strip() if adv else "В источниках преимущество явно не сформулировано — нужна оценка эксперта.",
        "advantage_source": adv[1]["source_name"] if adv else None,
        "case": {"title": lead["title"], "url": lead["url"], "site": lead["source_name"], "date": lead.get("published_at"),
                 "text": f"{lead['title']} — {lead['source_name']}" + (f", {lead['published_at'][:4]}" if lead.get("published_at") else "")},
        "why": [],
    }


def _src_word(n: int) -> str:
    return "источнике" if n % 10 == 1 and n % 100 != 11 else "источниках"


def build_card(s: dict, rank: int | None, use_llm: bool, log: list | None) -> dict:
    docs = _ranked(s["docs"])
    texts = _offline_texts(s, docs)
    name, llm_model = s["tech_name"], "без LLM (выдержки из источников)"
    if use_llm:
        h, model = router.hypothesis(" · ".join(s["keyphrases"]) or name, docs, log)
        if h:
            llm_model, name = model, h.get("name_ru") or name
            texts.update(description=h.get("description_ru") or texts["description"], quote=None,
                         advantage=h.get("advantage_ru") or texts["advantage"], advantage_source=None)
            texts["case"]["text"] = h.get("case_example") or texts["case"]["text"]
            if h.get("why_weak_ru"):
                texts["why"].append(h["why_weak_ru"])
    top_docs = docs[:5]
    summaries = [router.summarize_ru(d["title"], d.get("snippet") or "", log) if use_llm and d["lang"] != "ru"
                 else (None, None) for d in top_docs]
    chips = _key_chips(s)
    why = texts["why"] + [s["status_reason"]]
    why += [LEXICON_RU[x["feature"]] for x in s["ml_top"] if x["contribution"] > 0][:2]
    why += [p["detail"][:1].upper() + p["detail"][1:] for p in s["rule_parts"] if p["label"] in ("Свежесть", "Научная база")]
    return {
        "rank": rank, "name": name, "keyphrases": s["keyphrases"],
        "score": s["score"], "score_pct": round(s["score"] * 100), "confidence_ru": confidence_ru(s["score"]),
        "status": s["status"], "status_ru": STATUS_RU[s["status"]], "status_reason": s["status_reason"],
        "chips": chips, "description": texts["description"], "quote": texts["quote"],
        "advantage": texts["advantage"], "advantage_source": texts["advantage_source"], "case": texts["case"],
        "why": why, "predictors": _predictors(s),
        "score_parts": {"rules": s["rule_score"], "ml": s["ml_proba"],
                        "formula": f"{1 - ML_WEIGHT:.1f} × правила + {ML_WEIGHT:.1f} × модель"
                        if s["ml_proba"] is not None else "только правила (модель не обучена)"},
        "sources": [_source_card(d, *sm) for d, sm in zip(top_docs, summaries)],
        "n_sources": len(docs), "llm_model": llm_model, "built_at": date.today().isoformat(),
    }
