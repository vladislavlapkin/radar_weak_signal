"""Разбор открытого запроса: выделяем предметную область и переводим её на английский.

arXiv и OpenAlex ищут по-английски, поэтому русский запрос туда отправлять нельзя.
Порядок: YandexGPT Lite -> словарь предметных областей -> межъязыковые ссылки Википедии.
Способ перевода возвращается вместе с результатом и показывается в интерфейсе.
"""
import re
import httpx
from llm.router import expand_query

# служебные слова запроса — не предметная область
BOILERPLATE = [r"слаб\w* сигнал\w*", r"сигнал\w*", r"перспективн\w*", r"решени\w*", r"технолог\w*",
               r"тренд\w*", r"направлени\w*", r"зарождающ\w*", r"нов\w*", r"инновац\w*", r"развити\w*",
               r"будущ\w*", r"в области", r"в сфере", r"в отрасли", r"\bдля\b", r"\bв\b", r"\bна\b", r"\bи\b",
               r"\bпо\b", r"\bо\b"]
# от частного к общему: совпавший фрагмент вырезается, чтобы общий термин не сработал повторно
GLOSSARY = [
    (r"защит\w* (ии|искусственн\w* интеллект\w*)|безопасност\w* ии", "AI security"),
    (r"(промышленн|индустриальн)\w* (ии|искусственн\w* интеллект\w*)", "industrial artificial intelligence"),
    (r"инфраструктур\w* (ии|искусственн\w* интеллект\w*)", "AI infrastructure"),
    (r"больш\w* языков\w* модел\w*|\bllm\b", "large language models"),
    (r"генеративн\w*", "generative AI"),
    (r"ии-агент\w*|агент\w*", "AI agents"),
    (r"машинн\w* обучени\w*", "machine learning"),
    (r"компьютерн\w* зрени\w*", "computer vision"),
    (r"кибербезопасн\w*|информационн\w* безопасн\w*", "cybersecurity"),
    (r"\bии\b|искусственн\w* интеллект\w*|нейросет\w*", "artificial intelligence"),
    (r"финтех\w*|финансов\w* технолог\w*", "fintech"),
    (r"платеж\w*", "digital payments"),
    (r"антифрод\w*|мошенничеств\w*", "fraud detection"),
    (r"кредит\w*|скоринг\w*", "credit scoring"),
    (r"страхов\w*", "insurtech"),
    (r"банк\w*", "banking technology"),
    (r"блокчейн\w*|распределенн\w* реестр\w*", "blockchain"),
    (r"робот\w*", "robotics"),
    (r"квантов\w*", "quantum computing"),
    (r"периферийн\w* вычислени\w*|\bedge\b", "edge computing"),
    (r"интернет\w* вещей|\biot\b", "internet of things"),
    (r"полупроводник\w*|микроэлектрон\w*|чип\w*", "semiconductors"),
    (r"облачн\w*", "cloud computing"),
    (r"цод\w*|центр\w* обработки данных", "data centers"),
    (r"беспилотн\w*|автономн\w* транспорт\w*", "autonomous vehicles"),
    (r"дрон\w*|бпла", "drones"),
    (r"космо\w*|спутник\w*", "space technology"),
    (r"водород\w*", "hydrogen energy"),
    (r"аккумулятор\w*|батаре\w*|накопител\w* энерг\w*", "battery technology"),
    (r"энергетик\w*|энергосистем\w*", "energy systems"),
    (r"нефт\w*|газов\w*", "oil and gas technology"),
    (r"биотехнолог\w*", "biotechnology"),
    (r"медицин\w*|здравоохранени\w*", "healthcare technology"),
    (r"биометри\w*|идентификаци\w*|аутентификаци\w*", "digital identity"),
    (r"приватност\w*|конфиденциальн\w*", "privacy-preserving computation"),
    (r"логистик\w*", "logistics technology"),
    (r"ритейл\w*|розничн\w* торговл\w*", "retail technology"),
    (r"образовани\w*", "edtech"),
    (r"агро\w*|сельск\w* хозяйств\w*", "agritech"),
    (r"климат\w*|углерод\w*", "climate tech"),
    (r"материал\w*", "advanced materials"),
    (r"связ\w*|\b[56]g\b", "wireless communication"),
]
UA = {"User-Agent": "WeakSignalsRadar/1.0 (hackathon demo)"}


def _core(q: str) -> str:
    t = q.lower().replace("ё", "е")
    for p in BOILERPLATE:
        t = re.sub(p, " ", t)
    return re.sub(r"\s+", " ", t).strip(" ,.;:-")


def _glossary(text: str) -> list[str]:
    t, found = text.lower().replace("ё", "е"), []
    for pattern, en in GLOSSARY:
        if re.search(pattern, t):
            found.append(en)
            t = re.sub(pattern, " ", t)
    return found[:2]


def _wikipedia(term: str) -> str | None:
    """Русская статья Википедии по термину -> название её английской версии."""
    try:
        s = httpx.get("https://ru.wikipedia.org/w/api.php", headers=UA, timeout=8.0, params={
            "action": "query", "list": "search", "srsearch": term, "srlimit": 1, "format": "json"}).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        p = httpx.get("https://ru.wikipedia.org/w/api.php", headers=UA, timeout=8.0, params={
            "action": "query", "prop": "langlinks", "lllang": "en", "titles": hits[0]["title"],
            "format": "json"}).json()
        for page in p.get("query", {}).get("pages", {}).values():
            for ll in page.get("langlinks", []):
                return re.sub(r"\s*\(.*?\)", "", ll.get("*", "")).strip() or None
    except Exception:
        return None
    return None


def translate_query(q: str, log: list | None = None) -> dict:
    """{'ru': предметная область, 'en': запрос для arXiv/OpenAlex, 'method': как получен перевод}."""
    q = q.strip()
    if not re.search(r"[а-яё]", q, re.I):
        return {"ru": q, "en": q, "method": "запрос уже на английском"}
    llm = expand_query(q, log)
    if llm:
        return {"ru": llm.get("ru") or _core(q) or q, "en": llm["en"], "method": "YandexGPT Lite 5"}
    core = _core(q) or q
    terms = _glossary(core)
    if terms:
        return {"ru": core, "en": " ".join(terms), "terms": terms, "method": "словарь предметных областей"}
    wiki = _wikipedia(core)
    if wiki:
        return {"ru": core, "en": wiki, "method": "Википедия (межъязыковая ссылка)"}
    return {"ru": core, "en": core, "method": "перевод не найден — ищем как есть"}
