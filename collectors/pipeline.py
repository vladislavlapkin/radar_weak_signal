"""Пайплайн: параллельный сбор -> дедупликация -> кластеризация в технологии-кандидаты."""
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from . import arxiv_collector, openalex_collector, ddg_collector

RU_STOP = set("""а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его
ее если есть еще же за здесь и из или им их к как ко когда кто ли либо между меня мне может мы на над надо наш не него
нее нет ни них но ну о об однако он она они оно от очень по под после при про с со так также такой там те тем то того
тоже только том тот у уже чем что чтобы эта эти это этот я году года лет может могут можно который которые которых
которая которое сегодня статье статья новости компании компания рынка рынок""".split())
GENERIC = {"paper", "propose", "proposed", "approach", "method", "methods", "results", "result", "study", "based",
           "using", "used", "use", "new", "novel", "also", "show", "shows", "present", "work", "framework",
           "article", "research", "analysis", "existing", "different", "various", "provide", "provides", "however",
           "review", "systematic", "literature", "survey", "challenges", "challenge", "exploring", "explore",
           "towards", "toward", "evidence", "insights", "perspective", "perspectives", "impact", "role", "case",
           "guidance", "mechanisms", "test", "adaptive", "effects", "effect", "study", "overview", "future",
           "trends", "emerging", "current", "state", "art", "key", "need", "enhancing", "improving", "via",
           "launches", "raises", "announces", "report", "today", "company", "companies", "said", "says",
           "какие", "новые", "новой", "тренды", "тренд", "эксперты", "назвали", "будущее", "реальности"}
TITLE_CUT = re.compile(r"\s*(?::|\s[—–-]\s|\s\|\s)\s*")
SOURCES = {"arxiv": "arXiv", "openalex": "OpenAlex", "web_ru": "Веб (RU)", "news_en": "Новости (EN)"}


def collect(tr: dict, limit_each: int = 15) -> tuple[list[dict], dict]:
    """tr — результат collectors.query.translate_query. Возвращает документы и счётчики по источникам."""
    terms = tr.get("terms") or [tr["en"]]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            "arxiv": ex.submit(arxiv_collector.search, terms, limit_each),
            "openalex": ex.submit(openalex_collector.search, tr["en"], limit_each),
            "web_ru": ex.submit(ddg_collector.search, f"{tr['ru']} новые технологии {date.today().year}", limit_each),
            "news_en": ex.submit(ddg_collector.news, f"{tr['en']} startup", 10),
        }
        found = {k: f.result() for k, f in futures.items()}
    seen, docs = set(), []
    for batch in found.values():
        for d in batch:
            dd = d.to_dict()
            key = re.sub(r"\W+", " ", dd["title"].lower()).strip()
            if not dd["url"] or dd["url"] in seen or key in seen:
                continue
            seen.update({dd["url"], key})
            docs.append(dd)
    return docs, {SOURCES[k]: len(v) for k, v in found.items()}


def _keyphrases(centroid: np.ndarray, vocab: np.ndarray, banned: set[str], k: int = 3) -> list[str]:
    """Самые весомые фразы кластера для тегов: биграммы, без слов запроса и общих слов."""
    picked = []
    for i in np.argsort(-centroid)[:80]:
        term = vocab[i]
        if centroid[i] <= 0 or " " not in term or any(w in banned for w in term.split()):
            continue
        if not any(term in p or p in term for p in picked):
            picked.append(term)
        if len(picked) == k:
            break
    return picked


def clean_title(title: str, max_words: int = 10) -> str:
    """Заголовок документа -> название темы: без артикля, до двоеточия/тире, не длиннее max_words слов."""
    t = re.sub(r"^(a|an|the)\s+", "", " ".join(title.split()), flags=re.I)
    head = TITLE_CUT.split(t)[0]
    t = head if len(head.split()) >= 2 else t
    words = t.split()
    t = (" ".join(words[:max_words]) + ("…" if len(words) > max_words else "")).rstrip(" .,;")
    return t[:1].upper() + t[1:]


def cluster_to_candidates(docs: list[dict], query_terms: list[str], max_clusters: int = 24) -> list[dict]:
    """KMeans по TF-IDF; кандидат = кластер, названный по ключевым фразам."""
    if not docs:
        return []
    texts = [f"{d['title']}. {d['title']}. {d['snippet'] or ''}" for d in docs]
    stop = list(ENGLISH_STOP_WORDS | RU_STOP | GENERIC)
    banned = {w for t in query_terms for w in t.lower().split()} | set(stop)
    k = max(1, min(max_clusters, len(docs) // 3 or 1))
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words=stop, min_df=1, sublinear_tf=True,
                          token_pattern=r"(?u)\b[^\W\d_][\w-]{2,}\b")
    try:
        X = vec.fit_transform(texts)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        vocab = vec.get_feature_names_out()
    except ValueError:
        X, labels, vocab = None, [0] * len(docs), None
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)
    cands = []
    for lab, idx in groups.items():
        g = [docs[i] for i in idx]
        if X is not None:
            centroid = np.asarray(X[idx].mean(axis=0)).ravel()
            central = idx[int(np.argmax(X[idx] @ centroid))]  # документ, ближайший к центру кластера
            phrases = _keyphrases(centroid, vocab, banned)
        else:
            central, phrases = idx[0], []
        name = clean_title(docs[central]["title"])
        cands.append({"cluster_id": lab, "docs": g, "keyphrases": phrases, "tech_name": name,
                      "combined_text": " ".join(f"{x['title']}. {x['snippet'] or ''}" for x in g)[:6000]})
    return cands
