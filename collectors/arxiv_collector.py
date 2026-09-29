"""arXiv — препринты, бесплатно и без ключа.
Ищем точную фразу (или пересечение фраз) за последние два года, сортировка по релевантности:
слабый сигнал — свежая, но предметная публикация, а не просто последняя по дате.
"""
from datetime import date
import httpx, feedparser
from .base import SourceDoc
from scoring.trust import trust_for


def build_query(terms: list[str], years: int = 2) -> str:
    phrases = " AND ".join(f'all:"{t}"' for t in terms if t)
    today = date.today()
    start = today.replace(year=today.year - years)
    return f"({phrases}) AND submittedDate:[{start:%Y%m%d}0000 TO {today:%Y%m%d}2359]"


def search(terms: list[str], limit: int = 15) -> list[SourceDoc]:
    try:
        r = httpx.get("https://export.arxiv.org/api/query", timeout=20.0, follow_redirects=True,
                      params={"search_query": build_query(terms), "start": 0, "max_results": limit,
                              "sortBy": "relevance", "sortOrder": "descending"})
        feed = feedparser.parse(r.text)
        lvl, sc = trust_for("paper")
        return [SourceDoc(title=" ".join(e.get("title", "").split())[:300], url=e.get("link", ""),
                          source_name="arXiv", source_type="paper", lang="en",
                          snippet=" ".join(e.get("summary", "").split())[:1200],
                          published_at=(e.get("published", "")[:10] or None), trust_level=lvl, trust_score=sc)
                for e in feed.entries[:limit]]
    except Exception:
        return []
