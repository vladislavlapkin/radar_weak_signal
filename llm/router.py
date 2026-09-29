"""Роутер разрешённых LLM с явным логированием (требование ТЗ).
Lite — массовые задачи (перевод запроса, резюме источников), Pro — гипотезы по кандидатам.
Без доступа к API функции возвращают None, и вызывающий код строит экстрактивный текст —
демо не падает, а в выдаче честно указано, что текст собран без LLM.
"""
import hashlib, json, os, re, time
import httpx
from .prompts import QUERY_EXPAND, SUMMARIZE, HYPOTHESIS

URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
_status = {"checked": 0.0, "value": None}


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:12]


def _creds():
    return os.getenv("YANDEX_API_KEY", ""), os.getenv("YANDEX_FOLDER_ID", "")


def _models():
    return os.getenv("YANDEX_LITE_MODEL", "yandexgpt-lite/latest"), os.getenv("YANDEX_PRO_MODEL", "yandexgpt/rc")


def model_title(model: str) -> str:
    """Название модели для выдачи и логов: ТЗ требует раскрывать, какая модель ответила."""
    if model.startswith("yandexgpt-lite"):
        return "YandexGPT Lite 5"
    if model.startswith("yandexgpt"):
        return "YandexGPT Pro 5.1" if model.endswith("/rc") else "YandexGPT Pro 5"
    return model


def _call(prompt: str, model: str, max_tokens: int = 600, timeout: float = 40.0) -> tuple[str | None, int]:
    ak, folder = _creds()
    if not (ak and folder):
        return None, 0
    uri = f"gpt://{folder}/{model if '/' in model else model + '/latest'}"
    try:
        r = httpx.post(URL, headers={"Authorization": f"Api-Key {ak}", "x-folder-id": folder}, timeout=timeout,
                       json={"modelUri": uri, "messages": [{"role": "user", "text": prompt}],
                             "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": max_tokens}})
        if r.status_code != 200:
            return None, r.status_code
        return r.json()["result"]["alternatives"][0]["message"]["text"], 200
    except Exception:
        return None, -1


def status() -> dict:
    """Доступность LLM (кешируется на 5 минут, чтобы не делать десятки заведомо неудачных вызовов)."""
    if _status["value"] and time.time() - _status["checked"] < 300:
        return _status["value"]
    lite, pro = _models()
    ak, folder = _creds()
    if not (ak and folder):
        v = {"available": False, "model": "офлайн", "note": "Ключ YandexGPT не задан — тексты собраны из источников без LLM"}
    else:
        _, code = _call("ok", lite, max_tokens=5, timeout=15.0)
        if code == 200:
            v = {"available": True, "model": f"{model_title(lite)} + {model_title(pro)}", "note": ""}
        elif code == 403:
            v = {"available": False, "model": "офлайн",
                 "note": "YandexGPT: нет доступа (403) — сервисному аккаунту нужна роль ai.languageModels.user"}
        else:
            v = {"available": False, "model": "офлайн", "note": f"YandexGPT недоступна (код {code})"}
    _status.update(checked=time.time(), value=v)
    return v


def _json(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def _log(log: list | None, model: str, purpose: str, prompt: str, ok: bool):
    if log is not None:
        log.append({"model": model_title(model), "purpose": purpose, "prompt_hash": _hash(prompt), "ok": ok})


def expand_query(q: str, log: list | None = None) -> dict | None:
    if not status()["available"]:
        return None
    lite, _ = _models()
    prompt = QUERY_EXPAND.format(q=q)
    text, code = _call(prompt, lite, max_tokens=120)
    _log(log, lite, "query_expand", prompt, code == 200)
    d = _json(text)
    return d if d and d.get("en") else None


def summarize_ru(title: str, snippet: str, log: list | None = None) -> tuple[str | None, str | None]:
    """(резюме на русском, модель) или (None, None), если LLM недоступна."""
    if not (snippet and status()["available"]):
        return None, None
    lite, _ = _models()
    prompt = SUMMARIZE.format(title=title[:300], snippet=snippet[:1500])
    text, code = _call(prompt, lite, max_tokens=200)
    _log(log, lite, "summarize", prompt, code == 200)
    return (text.strip(), model_title(lite)) if text else (None, None)


def hypothesis(keyphrases: str, sources: list[dict], log: list | None = None) -> tuple[dict | None, str | None]:
    """Описание, преимущество, кейс и обоснование по источникам кластера (YandexGPT Pro)."""
    if not status()["available"]:
        return None, None
    _, pro = _models()
    src = "\n".join(f"[{i + 1}] {s.get('title', '')} ({s.get('published_at') or 'без даты'}): "
                    f"{(s.get('snippet') or '')[:400]}" for i, s in enumerate(sources[:6]))
    prompt = HYPOTHESIS.format(tech=keyphrases[:200], sources=src[:3500])
    text, code = _call(prompt, pro, max_tokens=700)
    _log(log, pro, "hypothesis", prompt, code == 200)
    d = _json(text)
    return (d, model_title(pro)) if d else (None, None)
