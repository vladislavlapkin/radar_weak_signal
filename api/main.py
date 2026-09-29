"""FastAPI: открытый поиск слабых сигналов, история запросов, карточки инсайтов."""
import json, time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from analysis.cards import build_card, score_candidate
from collectors.pipeline import cluster_to_candidates, collect
from collectors.query import translate_query
from db import store
from llm import router
from ml.inference import get_model

METRICS = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "metrics.json"


@asynccontextmanager
async def lifespan(_app):
    for _ in range(10):  # Postgres в compose может подняться позже API
        if store.init():
            break
        time.sleep(2)
    yield


app = FastAPI(title="Радар слабых сигналов", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchReq(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    top_k: int = Field(default=15, ge=1, le=30)


def model_info() -> dict:
    info = {"name": "Логистическая регрессия · 16 признаков", "loaded": get_model() is not None}
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
        info.update(cv_accuracy=m["cv_5x5"][m["model"]]["accuracy"]["mean"], cv_f1=m["cv_5x5"][m["model"]]["f1"]["mean"])
    return info


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    return {"llm": router.status(), "model": model_info(), "db": store.engine() is not None}


@app.post("/search")
def search(req: SearchReq):
    t0, llm_log = time.time(), []
    llm = router.status()
    tr = translate_query(req.query, llm_log)
    docs, by_source = collect(tr)
    scored = [score_candidate(c) for c in cluster_to_candidates(docs, tr.get("terms") or [tr["en"]])]
    kept = sorted([s for s in scored if s["status"] in ("weak", "low_trust")],
                  key=lambda s: (s["status"] == "weak", s["score"]), reverse=True)[:req.top_k]
    rejected = sorted([s for s in scored if s["status"] not in ("weak", "low_trust")], key=lambda s: -s["score"])
    with ThreadPoolExecutor(max_workers=6) as ex:  # LLM-обогащение только для ТОПа, параллельно
        signals = list(ex.map(lambda p: build_card(p[1], p[0] + 1, llm["available"], llm_log), enumerate(kept)))
    rejected_cards = [build_card(s, None, False, None) for s in rejected]
    for card in signals + rejected_cards:
        card["query"] = req.query
    by_status = {k: sum(1 for s in rejected if s["status"] == k) for k in ("mature_rejected", "noise_rejected")}
    result = {
        "search_id": None, "query": req.query, "translation": tr, "llm": llm, "model": model_info(),
        "stats": {"sources": len(docs), "by_source": by_source, "candidates": len(scored),
                  "signals": len(signals), "confident": sum(1 for c in signals if c["status"] == "weak" and c["score"] > 0.75),
                  "rejected": len(rejected), "rejected_mature": by_status["mature_rejected"],
                  "rejected_noise": by_status["noise_rejected"],
                  "low_trust": sum(1 for c in signals if c["status"] == "low_trust")},
        "signals": signals, "rejected": rejected_cards, "elapsed_sec": round(time.time() - t0, 1),
    }
    result["search_id"] = store.save_search(result, docs, llm_log)
    return result


@app.get("/history")
def history(limit: int = 12):
    return store.history(limit)


@app.get("/searches/{search_id}")
def get_search(search_id: int):
    found = store.get_search(search_id)
    if not found:
        raise HTTPException(404, "Запрос не найден")
    head, cards = found["head"], found["cards"]
    meta = head.get("meta") or {}
    return {"search_id": search_id, "query": head["query"], **meta, "elapsed_sec": head["elapsed_sec"],
            "signals": [c for c in cards if c["status"] in ("weak", "low_trust")],
            "rejected": [c for c in cards if c["status"] not in ("weak", "low_trust")],
            "created_at": head["created_at"].isoformat(timespec="minutes")}


@app.get("/signals/{signal_id}")
def get_signal(signal_id: int):
    card = store.get_signal(signal_id)
    if not card:
        raise HTTPException(404, "Сигнал не найден")
    return card
