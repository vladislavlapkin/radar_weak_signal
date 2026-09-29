"""Инференс: вероятность «слабый сигнал» + вклад признаков для объяснения."""
from pathlib import Path
import joblib
from .features import LEXICON_NAMES, LEXICON_RU, LEXICON_SHORT

MODEL_PATH = Path(__file__).parent / "artifacts" / "model.pkl"

_model = None


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def explain(full_text: str, top: int = 5) -> list[dict]:
    """Вклад признаков в логит: вес модели × стандартизированное значение признака.
    Показываем только сработавшие признаки, самые сильные по модулю."""
    m = get_model()
    if m is None or "lex" not in m.named_steps:
        return []
    raw = m.named_steps["lex"].transform([full_text])[0]
    scaled = m.named_steps["sc"].transform([raw])[0]
    contrib = scaled * m.named_steps["clf"].coef_[0]
    items = [{"feature": f, "short": LEXICON_SHORT[f], "value": float(raw[i]),
              "contribution": round(float(contrib[i]), 3),
              "text_ru": f"{LEXICON_RU[f]} ({'+' if contrib[i] >= 0 else '−'}{abs(contrib[i]):.2f})"}
             for i, f in enumerate(LEXICON_NAMES) if raw[i] > 0]
    return sorted(items, key=lambda x: -abs(x["contribution"]))[:top]


def predict_proba(full_text: str, num_feats: dict | None = None) -> tuple[float, str]:
    """num_feats оставлен для совместимости с api/main.py: модель работает только по тексту."""
    m = get_model()
    if m is None:
        return 0.5, "ML-модель не обучена (нет artifacts/model.pkl) — используется только rule-score"
    p = float(m.predict_proba([full_text])[0, 1])
    why = explain(full_text, top=3)
    reasons = "; ".join(x["text_ru"] for x in why) or "сработавших признаков нет"
    return p, f"ML proba={p:.2f} (логистическая регрессия на {len(LEXICON_NAMES)} признаках). Главное: {reasons}"
