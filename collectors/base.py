"""Нормализованный документ источника."""
from dataclasses import dataclass, asdict

@dataclass
class SourceDoc:
    title: str
    url: str
    source_name: str
    source_type: str
    lang: str
    snippet: str
    published_at: str | None = None
    trust_level: str = "low"
    trust_score: float = 0.3

    def to_dict(self): return asdict(self)
