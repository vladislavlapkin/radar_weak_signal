"""OpenAlex — рецензируемые статьи, бесплатно. Берём работы за последние три года
и восстанавливаем аннотацию из инвертированного индекса."""
from datetime import date
import httpx
from .base import SourceDoc
from scoring.trust import trust_for


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def search(query_en: str, limit: int = 15, years: int = 3) -> list[SourceDoc]:
    try:
        since = date.today().replace(year=date.today().year - years)
        r = httpx.get("https://api.openalex.org/works", timeout=20.0, params={
            "search": query_en, "per-page": limit, "sort": "relevance_score:desc",
            "filter": f"from_publication_date:{since:%Y-%m-%d},has_abstract:true",
            "mailto": "hackathon@example.com"})
        lvl, sc = trust_for("paper")
        out = []
        for w in r.json().get("results", [])[:limit]:
            doi = w.get("doi") or ""
            url = doi if doi.startswith("http") else (w.get("primary_location") or {}).get("landing_page_url") or ""
            venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "OpenAlex"
            out.append(SourceDoc(title=(w.get("title") or "Без названия")[:300], url=url or w.get("id", ""),
                                 source_name=venue[:120], source_type="paper", lang=(w.get("language") or "en")[:2],
                                 snippet=_abstract(w.get("abstract_inverted_index"))[:1200],
                                 published_at=str(w.get("publication_date") or "")[:10] or None,
                                 trust_level=lvl, trust_score=sc))
        return out
    except Exception:
        return []
