"""Веб-поиск без ключа (библиотека ddgs): первичные индикаторы и отраслевые медиа.
text() — русскоязычный веб по исходному запросу, news() — англоязычные новости за год (с датами)."""
from .base import SourceDoc
from scoring.trust import trust_for

TYPE_RULES = [  # (тип источника, подстроки домена/URL) — первое совпадение
    ("gov", [".gov", "gov.ru", "gov.uk", "europa.eu", "cbr.ru", "digital.gov.ru", "rosstat", "oecd.org", "un.org"]),
    ("edu", [".edu", "ac.ru", "ac.uk", "university", "universit", "msu.ru", "hse.ru", "skoltech", "mit.edu"]),
    ("paper", ["arxiv", "cyberleninka", "elibrary", "springer", "sciencedirect", "nature.com", "doi.org", "mdpi"]),
    ("conf", ["ieee.org", "acm.org", "neurips", "usenix"]),
    ("patent", ["patent", "espacenet", "fips.ru", "wipo.int"]),
    ("analytics", ["gartner", "mckinsey", "cbinsights", "idc.com", "forrester", "tadviser", "pwc.", "deloitte"]),
    ("press", ["prnewswire", "businesswire", "globenewswire", "press-release", "пресс-релиз"]),
    ("industry_media", ["habr", "vc.ru", "cnews", "rbc.ru", "kommersant", "forbes", "siliconangle", "techcrunch",
                        "venturebeat", "securityweek", "theregister", "wired", "reuters", "bloomberg", "zdnet",
                        "therecord", "darkreading", "finextra", "pymnts", "iot-analytics", "therobotreport"]),
    ("blog", ["blog", "medium.com", "dzen", "substack", "teletype"]),
    ("social", ["vk.com", "t.me", "twitter", "x.com", "reddit", "youtube", "linkedin"]),
]


def guess_type(url: str) -> str:
    u = url.lower()
    for kind, needles in TYPE_RULES:
        if any(n in u for n in needles):
            return kind
    return "aggregator"


def guess_lang(text: str) -> str:
    ru = sum(1 for c in text if "а" <= c.lower() <= "я" or c in "ёЁ")
    return "ru" if ru > 5 else "en"


def _doc(url: str, title: str, body: str, date: str | None = None, site: str | None = None) -> SourceDoc:
    kind = guess_type(url + " " + title)
    lvl, sc = trust_for(kind)
    host = url.split("/")[2].removeprefix("www.") if "://" in url else url
    return SourceDoc(title=title[:300], url=url, source_name=(site or host)[:120], source_type=kind,
                     lang=guess_lang(title + body), snippet=body[:800], published_at=date,
                     trust_level=lvl, trust_score=sc)


def search(query: str, limit: int = 20) -> list[SourceDoc]:
    try:
        from ddgs import DDGS
        return [_doc(r.get("href", ""), r.get("title", ""), r.get("body", ""))
                for r in DDGS().text(query, max_results=limit)]
    except Exception:
        return []


def news(query: str, limit: int = 10) -> list[SourceDoc]:
    try:
        from ddgs import DDGS
        return [_doc(r.get("url", ""), r.get("title", ""), r.get("body", ""),
                     date=(r.get("date") or "")[:10] or None, site=r.get("source"))
                for r in DDGS().news(query, max_results=limit, timelimit="y")]
    except Exception:
        return []
