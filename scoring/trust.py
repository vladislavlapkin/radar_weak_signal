"""Карта типов источников -> доверенность. По ТЗ."""
TRUST_MAP = {
    "gov": ("high", 1.0),
    "edu": ("high", 1.0),
    "patent": ("high", 1.0),
    "paper": ("high", 1.0),
    "conf": ("high", 1.0),
    "registry": ("high", 1.0),
    "industry_media": ("medium", 0.6),
    "analytics": ("medium", 0.6),
    "company": ("medium", 0.6),
    "blog": ("low", 0.3),
    "social": ("low", 0.3),
    "aggregator": ("low", 0.3),
    "press": ("low", 0.3),
}

def trust_for(source_type: str):
    return TRUST_MAP.get(source_type, ("low", 0.3))

TYPE_RU = {"gov": "Госорган", "edu": "Университет", "patent": "Патент", "paper": "Научная статья",
           "conf": "Конференция", "registry": "Реестр", "industry_media": "Отраслевое СМИ",
           "analytics": "Аналитический отчёт", "company": "Сайт компании", "blog": "Блог", "social": "Соцсеть",
           "aggregator": "Агрегатор", "press": "Пресс-релиз"}
TRUST_RU = {"high": "Высокое", "medium": "Среднее", "low": "Пониженное"}
