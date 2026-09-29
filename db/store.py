"""Хранение в PostgreSQL: запросы, сырые документы, карточки сигналов, журнал LLM.
Если база недоступна, поиск всё равно работает — сохранение пропускается с предупреждением в логе."""
import json, logging, os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger("store")
SCHEMA = Path(__file__).with_name("schema.sql")
_engine: Engine | None = None


def engine() -> Engine | None:
    global _engine
    url = os.getenv("DATABASE_URL")
    if _engine is None and url:
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5)
    return _engine


def init() -> bool:
    eng = engine()
    if eng is None:
        return False
    try:
        with eng.begin() as c:
            c.exec_driver_sql(SCHEMA.read_text(encoding="utf-8"))
        return True
    except Exception as e:  # БД ещё поднимается или недоступна
        log.warning("schema init failed: %s", e)
        return False


def save_search(result: dict, docs: list[dict], llm_calls: list[dict]) -> int | None:
    """Сохраняет выдачу целиком и проставляет id сигналов в карточки. Возвращает id поиска."""
    eng = engine()
    if eng is None:
        return None
    try:
        with eng.begin() as c:
            s = result["stats"]
            sid = c.execute(text("""
                INSERT INTO searches (query, query_en, translation_method, n_sources, n_candidates, n_signals,
                                      n_confident, n_rejected, llm_model, elapsed_sec, meta)
                VALUES (:q, :en, :m, :ns, :nc, :nsig, :conf, :rej, :llm, :el, CAST(:meta AS JSONB)) RETURNING id"""),
                dict(q=result["query"], en=result["translation"]["en"], m=result["translation"]["method"],
                     ns=s["sources"], nc=s["candidates"], nsig=s["signals"], conf=s["confident"],
                     rej=s["rejected"], llm=result["llm"]["model"], el=result["elapsed_sec"],
                     meta=json.dumps({k: result[k] for k in ("translation", "stats", "llm", "model")},
                                     ensure_ascii=False))).scalar_one()
            doc_ids = {}
            for d in docs:
                doc_ids[d["url"]] = c.execute(text("""
                    INSERT INTO documents (query, title, url, published_at, source_name, source_type, lang,
                                           trust_level, trust_score, snippet, search_id)
                    VALUES (:q, :title, :url, :pub, :src, :type, :lang, :tl, :ts, :snip, :sid)
                    ON CONFLICT (url) DO UPDATE SET search_id = EXCLUDED.search_id, snippet = EXCLUDED.snippet
                    RETURNING id"""),
                    dict(q=result["query"], title=d["title"][:1000], url=d["url"], pub=d.get("published_at"),
                         src=d["source_name"], type=d["source_type"], lang=d["lang"], tl=d["trust_level"],
                         ts=d["trust_score"], snip=d.get("snippet"), sid=sid)).scalar_one()
            for card in result["signals"] + result["rejected"]:
                card["id"] = c.execute(text("""
                    INSERT INTO signals (query, tech_name, description_ru, advantage_ru, case_example, weak_score,
                                         ml_proba, rule_score, predictors, maturity_verdict, reject_reason, llm_model,
                                         source_ids, search_id, rank, card)
                    VALUES (:q, :name, :descr, :adv, :case, :score, :ml, :rule, CAST(:preds AS JSONB), :verdict,
                            :reason, :llm, :src, :sid, :rank, CAST(:card AS JSONB)) RETURNING id"""),
                    dict(q=result["query"], name=card["name"], descr=card["description"], adv=card["advantage"],
                         case=card["case"]["text"], score=card["score"], ml=card["score_parts"]["ml"],
                         rule=card["score_parts"]["rules"], preds=json.dumps(card["predictors"], ensure_ascii=False),
                         verdict=card["status"], reason=card["status_reason"], llm=card["llm_model"],
                         src=[doc_ids[s["url"]] for s in card["sources"] if s["url"] in doc_ids],
                         sid=sid, rank=card.get("rank"), card="{}")).scalar_one()
                card["search_id"] = sid
                c.execute(text("UPDATE signals SET card = CAST(:card AS JSONB) WHERE id = :id"),
                          dict(card=json.dumps(card, ensure_ascii=False), id=card["id"]))
            for call in llm_calls:
                c.execute(text("INSERT INTO llm_log (model, purpose, prompt_hash, ok, search_id) "
                               "VALUES (:m, :p, :h, :ok, :sid)"),
                          dict(m=call["model"], p=call["purpose"], h=call["prompt_hash"], ok=call["ok"], sid=sid))
        return sid
    except Exception as e:
        log.warning("save_search failed: %s", e)
        return None


def history(limit: int = 12) -> list[dict]:
    eng = engine()
    if eng is None:
        return []
    try:
        with eng.connect() as c:
            rows = c.execute(text("""SELECT id, query, n_sources, n_signals, n_confident, created_at
                                     FROM searches ORDER BY created_at DESC LIMIT :n"""), {"n": limit}).mappings()
            return [{**r, "created_at": r["created_at"].isoformat(timespec="minutes")} for r in rows]
    except Exception as e:
        log.warning("history failed: %s", e)
        return []


def get_search(search_id: int) -> dict | None:
    """Восстанавливает выдачу из базы: шапка запроса + карточки в исходном порядке."""
    eng = engine()
    if eng is None:
        return None
    with eng.connect() as c:
        head = c.execute(text("SELECT * FROM searches WHERE id = :id"), {"id": search_id}).mappings().first()
        if not head:
            return None
        cards = [r[0] for r in c.execute(text(
            "SELECT card FROM signals WHERE search_id = :id ORDER BY rank NULLS LAST, weak_score DESC"),
            {"id": search_id})]
    return {"head": dict(head), "cards": cards}


def get_signal(signal_id: int) -> dict | None:
    eng = engine()
    if eng is None:
        return None
    with eng.connect() as c:
        row = c.execute(text("SELECT card FROM signals WHERE id = :id"), {"id": signal_id}).first()
    return row[0] if row else None
