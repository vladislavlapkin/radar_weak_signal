"""Фильтр мейнстрима, хайпа и шума. Возвращает вердикт и причину на русском.
Работает в два слоя: словари маркеров (правила) и признаки обученной модели."""
import re

MATURITY = {  # подпись для выдачи: паттерн
    "стандарт ISO/IEC/ГОСТ": r"\bISO\b|\bIEC\b|ГОСТ",
    "отраслевой стандарт": r"отраслев\w* стандарт|industry standard|de facto standard|стандарт де-факто",
    "лидеры рынка": r"лидер\w* рынка|market leader|dominant player",
    "доля рынка": r"доля рынка|market share",
    "массовое внедрение": r"массов\w* внедрен|mass adoption|widely adopted|mainstream",
    "Magic Quadrant": r"Gartner (Magic )?Quadrant",
}
HYPE = {
    "«революционный»": r"революци|revolutionary",
    "«убийца X»": r"убийца |killer",
    "«game changer»": r"game.?changer",
    "гарантии результата": r"гарантирован|guaranteed",
    "«хайп»": r"хайп|\bhype\b",
}
MODEL_REASONS = {  # признак модели -> вердикт и формулировка
    "mass_adoption": ("mature_rejected", "модель видит признаки массового внедрения"),
    "standards": ("mature_rejected", "модель видит отраслевые стандарты и регулирование"),
    "leaders": ("mature_rejected", "модель видит выраженных лидеров рынка"),
    "decline": ("mature_rejected", "модель видит стабилизацию или спад интереса"),
    "not_technology": ("noise_rejected", "это не технология, а подборка, рейтинг или прогноз рынка"),
    "hype": ("noise_rejected", "модель видит хайп-маркеры без подтверждения"),
    "marketing_only": ("noise_rejected", "источники похожи на рекламу и пресс-релизы"),
    "no_evidence": ("noise_rejected", "нет подтверждения публикациями или прототипом"),
}
# Заголовки-«не технологии»: подборки, ленты, прогнозы (класс «Шум» из разметки негативов)
NOISE_TITLE = re.compile(r"^(топ|лучшие|тренды|новости|главные)\b|последние (и свежие )?новости|новости сегодня|"
                         r"тренд\w* .{0,40}20\d\d|\b20\d\d\b.{0,20}тренд|\btop \d+|best .{0,30}\d{4}|"
                         r"trends (in|for|to watch)|прогноз\w* рынка|market (size|forecast)|\bрейтинг\b|"
                         r"тенденци\w* .{0,40}20\d\d|\d+ (главн|крупнейш|ключев|важн)\w* (тренд|тенденц)|"
                         r"^[\w\s,-]{3,40}\s(в|in)\s20\d\d(\s(году|год))?$|что ждет|what to expect", re.I)
REVIEW_TITLE = re.compile(r"literature review|systematic review|bibliometric|scoping review|state of the art|"
                          r"обзор литературы|систематический обзор", re.I)
STATUS_RU = {"weak": "Слабый сигнал", "mature_rejected": "Исключён: зрелая технология",
             "noise_rejected": "Исключён: шум или хайп", "low_trust": "Пониженная достоверность"}


def markers(text: str) -> tuple[list[str], list[str]]:
    t = text or ""
    return ([k for k, p in MATURITY.items() if re.search(p, t, re.I)],
            [k for k, p in HYPE.items() if re.search(p, t, re.I)])


def maturity_check(text: str, n_sources: int, high_share: float, low_only: bool,
                   ml_proba: float | None = None, ml_top: list[dict] | None = None,
                   titles: list[str] | None = None) -> tuple[str, str]:
    mature, hype = markers(text)
    titles = titles or []
    if titles and sum(bool(NOISE_TITLE.search(t)) for t in titles) * 2 > len(titles):
        return "noise_rejected", "Это не технология: подборка трендов, лента новостей или прогноз рынка"
    if titles and all(REVIEW_TITLE.search(t) for t in titles):
        return "mature_rejected", "Только обзорные статьи: тема уже сложилась настолько, что её систематизируют"
    if mature:
        return "mature_rejected", f"Зрелая технология: в источниках есть {', '.join(mature[:3])}"
    if ml_proba is not None and ml_proba < 0.35 and ml_top:
        negative = [x for x in ml_top if x["contribution"] < 0 and x["feature"] in MODEL_REASONS]
        if negative:
            verdict, why = MODEL_REASONS[negative[0]["feature"]]
            return verdict, f"Модель уверена, что это не слабый сигнал ({ml_proba:.0%}): {why}"
    if low_only:
        return "low_trust", "Только блоги, соцсети и агрегаторы — нужно подтверждение независимым источником"
    if n_sources <= 1 and high_share == 0:
        return "noise_rejected", "Единственный источник без научного подтверждения — похоже на информационный шум"
    if hype and high_share == 0:
        return "noise_rejected", f"Хайп без научной базы: {hype[0]}"
    return "weak", "Ранняя стадия: есть свежие публикации и рост интереса, признаков зрелого рынка нет"
