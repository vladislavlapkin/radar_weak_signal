"""Признаки для ML + инференса из открытого поиска. 12 числовых + текст."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

TEXT_COL_CANDIDATES = ["description", "text", "abstract", "title_description", "name_description"]
MAT_KW = ["iso", "гост", "ieee", "лидер рынка", "доля рынка", "массовое внедрение",
          "отраслевой стандарт", "market leader", "mass adoption", "industry standard"]
HYPE_KW = ["революци", "убийца", "хайп", "revolutionary", "game-changer", "disrupt"]
GRANT_KW = ["грант", "grant", "seed", "посевн", "trl", "рнф", "rsf"]

def pick_text_column(df):
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    # иначе склеить все object-колонки
    obj = [c for c in df.columns if df[c].dtype == object]
    return obj[0] if obj else df.columns[0]

def numeric_features_from_meta(total=1, growth=0.0, patents=0, patent_growth=0.0,
                               n_sources=1, n_types=1, n_authors=1, age_days=365,
                               text="", science_share=0.0, trust_mean=0.5, p95=500):
    t = (text or "").lower()
    mat = sum(1 for k in MAT_KW if k in t) / 3.0
    hype = sum(1 for k in HYPE_KW if k in t) / 3.0
    grants = sum(1 for k in GRANT_KW if k in t)
    diversity = min(1.0, (n_types / 4.0 + min(n_sources, 10) / 10.0) / 2.0)
    author_disp = min(1.0, n_authors / 8.0)
    return {
        "total_mentions": float(total), "growth_12m": float(growth),
        "patent_count": float(patents), "patent_growth": float(patent_growth),
        "source_diversity": float(diversity), "age_days": float(age_days),
        "maturity_kw": float(min(1.0, mat)), "hype_kw": float(min(1.0, hype)),
        "grant_mentions": float(min(3, grants)), "author_dispersion": float(author_disp),
        "science_share": float(science_share), "trust_mean": float(trust_mean),
        # для rule-скоринга
        "maturity_penalty": float(min(1.0, mat)), "hype_penalty": float(min(1.0, hype)),
        "n_sources": int(n_sources), "n_types": int(n_types), "p95": float(p95),
    }

NUM_COLS = ["total_mentions", "growth_12m", "patent_count", "patent_growth",
            "source_diversity", "age_days", "maturity_kw", "hype_kw",
            "grant_mentions", "author_dispersion", "science_share", "trust_mean"]


# --- Интерпретируемые признаки-критерии для ML-модели (обучение на датасете и инференс) ---
# Каждый признак = критерий из ТЗ. Значение = число разных сработавших маркеров (0..3).
# Маркеры RU+EN: датасет русскоязычный, а открытый поиск приносит в основном EN-статьи.
# neg_to: если маркер стоит в отрицании («стандартов ещё нет», «нет публикаций»),
# он засчитывается в противоположный признак.
LEXICON = {
    "early_stage": dict(ru="Ранняя стадия: концепция, исследование, прототип, пилот", patterns=[
        r"концепц", r"исследован", r"прототип", r"\bpoc\b", r"proof.of.concept", r"пилот", r"\bpilot",
        r"testbed", r"лаборатор", r"доклинич", r"опытн\w* образ", r"stealth", r"\bmvp\b",
        r"prototype", r"feasibility", r"early.stage", r"laborator", r"preclinical"]),
    "early_adoption": dict(ru="Первые внедрения и ранние клиенты", patterns=[
        r"раннее внедрен", r"ранние внедрен", r"ранн\w* (пилот|сери|поставк|клиент)",
        r"перв\w* (внедрен|клиент|интеграц|поставк|полис|коммерч|кластер|программ|сделк|продаж)",
        r"мелкосерийн", r"early adopt", r"first (deployment|customer|commercial)"]),
    "science": dict(ru="Научная база: статьи, препринты, патенты, конференции", neg_to="no_evidence", patterns=[
        r"препринт", r"preprint", r"arxiv", r"\bnature\b", r"\bscience\b", r"журнал", r"патент", r"patent",
        r"публикац", r"конференц", r"neurips|icml|iclr|cvpr|acl\b", r"академ", r"университет", r"universit",
        r"\bdoi\b", r"peer.review", r"рецензир", r"journal", r"conference", r"we propose"]),
    "early_funding": dict(ru="Ранние инвестиции: гранты, seed, Series A, стартапы", patterns=[
        r"pre-?seed", r"\bseed\b", r"посевн", r"series a\b", r"раунд\w* a\b", r"грант", r"\bgrant",
        r"венчур", r"venture", r"стартап", r"startup", r"\$\s?\d+(\.\d+)?\s?m\b", r"млн \$|\$\d+.?млн"]),
    "late_market": dict(ru="Зрелый бизнес: IPO, поглощения, миллионы пользователей", patterns=[
        r"\bipo\b", r"series [d-f]\b", r"поглощ", r"acquired|acquisition", r"публичн\w* компан",
        r"капитализац", r"млрд (операц|пользоват|устройств)", r"миллиард\w* (операц|пользоват|устройств)",
        r"сотни миллионов", r"миллион\w* (пользоват|устройств|сайтов|операц)", r"(million|billion)s? of users"]),
    "mass_adoption": dict(ru="Массовое внедрение, зрелость, товарная функция", neg_to="market_not_formed", patterns=[
        r"массов", r"повсеместн", r"mainstream", r"широко (распростран|использ|применя)", r"де-?факто",
        r"зрел", r"товарн\w* (функц|категор)", r"commodity", r"есть у всех|всеми крупными|у всех банков",
        r"большинств\w* (компаний|банков|предприятий|новых|корпоратив|крупных|карточн)",
        r"практически (в каждом|все)", r"стандартн\w* (практик|функц|услуг|оборудован|модул|инструмент|элемент|част|архитектур)",
        r"давно (серийн|в производств|известн|использ)",
        r"widely (adopted|used|deployed)", r"ubiquitous", r"well.established", r"\bmature\b"]),
    "standards": dict(ru="Отраслевые стандарты и регулирование", neg_to="market_not_formed", patterns=[
        r"стандарт(?!н)", r"\biso\b", r"гост", r"\bnist\b", r"\bieee\b", r"регламент", r"регулятор",
        r"регулиру", r"обязательн", r"закон", r"\bstandard", r"regulat", r"compliance"]),
    "leaders": dict(ru="Выраженные лидеры, рынок поделён", neg_to="market_not_formed", patterns=[
        r"лидер\w* рынка", r"выраженн\w* лидер", r"доминир", r"доля рынка", r"олигопол", r"гиперскейлер",
        r"поделен", r"консолидирован", r"крупнейш\w* производител", r"gartner", r"market (share|leader)",
        r"dominant", r"incumbent", r"hyperscaler"]),
    "market_not_formed": dict(ru="Рынок, категория и стандарты ещё не сформированы", patterns=[
        r"рын\w* (еще|пока) не", r"не сформирова", r"(еще|пока) нет", r"почти не обсужда", r"нишев",
        r"единичн", r"мало (покупател|игрок|компаний|упоминан|внедрен|клиент)", r"немногочисл",
        r"no (market|standard)", r"nascent", r"emerging"]),
    "growth": dict(ru="Рост упоминаний", patterns=[
        r"раст[её]т|растет", r"\bрост", r"растущ", r"увеличива", r"ускорени", r"growing", r"growth", r"increasing"]),
    "growth_fast": dict(ru="Быстрый, кратный рост", patterns=[
        r"быстр", r"кратн", r"скачкообразн", r"резк\w* рост", r"удво(и|е)", r"\bx\d", r"вырос\w* в \d",
        r"rapid", r"surge", r"doubl"]),
    "decline": dict(ru="Стабилизация или спад интереса", patterns=[
        r"стабильн", r"снижа", r"спад", r"падени|упал", r"угаса", r"стагнац", r"насыщ", r"плато",
        r"закрыт|закрыл", r"прекращ", r"сверну", r"снят\w* с (продаж|рынка)", r"уш(ел|ла|ли) с рынка", r"вытесн", r"устарев", r"потерял",
        r"declin", r"obsolete", r"legacy", r"deprecat", r"discontinu"]),
    "hype": dict(ru="Хайп-маркеры: «революция», гарантии, «убийца X»", patterns=[
        r"революц", r"убийц", r"заменит\w* (всех|все|любо|полностью)|полностью замен", r"game.?changer",
        r"\b1000\s?(x|%)", r"гарантир", r"изменит вс", r"прорыв век", r"в каждом (доме|смартфоне)",
        r"уже в следующем году", r"сознани", r"пик ожиданий", r"вечн\w* батаре", r"\bagi\b", r"revolution", r"disrupt"]),
    "marketing_only": dict(ru="Источники — реклама, пресс-релизы, соцсети", patterns=[
        r"пресс-релиз", r"соцсет", r"реклам", r"маркетинг", r"блогер", r"whitepaper", r"перепечат",
        r"развлекательн", r"инфлюенсер", r"press release", r"sponsored", r"advertis", r"marketing"]),
    "no_evidence": dict(ru="Нет подтверждения: публикаций, прототипа, независимых тестов", patterns=[
        r"без (публикац|прототип|подтвержд|независим|пилот|лиценз|техническ|коммерческ\w* внедрен)",
        r"нет (прототип|подтвержд|независим|лиценз|реальн|стат|вес|научн)", r"не подтвержд",
        r"только (заявлен|пресс-релиз|whitepaper|маркетинг)", r"unverified", r"no evidence"]),
    "not_technology": dict(ru="Не технология: подборка, рейтинг, прогноз рынка, каталог, публицистика", patterns=[
        r"не технолог", r"подборк", r"seo", r"прогноз\w* рынка|рын\w* .{0,25}вырастет", r"каталог",
        r"рубрик", r"публицистик", r"колонк\w* (о|об|про)", r"лучши\w* (нейросет|сервис|инструмент)",
        r"top.?\d+", r"listicle", r"market (forecast|size)"]),
}
LEXICON_NAMES = list(LEXICON)
LEXICON_RU = {k: v["ru"] for k, v in LEXICON.items()}
LEXICON_SHORT = {"early_stage": "Ранняя стадия", "early_adoption": "Первые внедрения", "science": "Научная база",
                 "early_funding": "Ранние инвестиции", "late_market": "Зрелый бизнес",
                 "mass_adoption": "Массовое внедрение", "standards": "Стандарты", "leaders": "Лидеры рынка",
                 "market_not_formed": "Рынок не сформирован", "growth": "Рост упоминаний",
                 "growth_fast": "Кратный рост", "decline": "Спад интереса", "hype": "Хайп",
                 "marketing_only": "Реклама и соцсети", "no_evidence": "Нет подтверждения",
                 "not_technology": "Не технология"}
_COMPILED = {k: [re.compile(p) for p in v["patterns"]] for k, v in LEXICON.items()}
_NEG = re.compile(r"\b(нет|не|без|ни|отсутств\w*|no|not|without|lack\w*)\b")
_SENT = re.compile(r"[.!?;\n]+")


def lexicon_features(text: str) -> dict:
    """Текст -> {признак: 0..3}. Отрицание ищется в том же предложении рядом с маркером."""
    t = (text or "").lower().replace("ё", "е")
    sents = _SENT.split(t)
    hits = {k: set() for k in LEXICON}
    for name, pats in _COMPILED.items():
        target = LEXICON[name].get("neg_to")
        for i, p in enumerate(pats):
            for s in sents:
                for m in p.finditer(s):
                    window = s[max(0, m.start() - 50): m.end() + 80]
                    negated = target is not None and _NEG.search(window)
                    hits[target if negated else name].add((name, i))
    return {k: float(min(3, len(v))) for k, v in hits.items()}


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """sklearn-трансформер: список текстов -> матрица признаков-критериев."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([list(lexicon_features(t).values()) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(LEXICON_NAMES)
