import { useEffect, useState } from "react";

const EXAMPLES = ["технологии в ИИ", "перспективные решения в финтехе", "слабые сигналы в кибербезопасности", "роботы"];

export const pad = (n) => String(n).padStart(2, "0");
export const cardKey = (card, index) => (card.id ? String(card.id) : `local-${index}`);
export const pct = (x) => `${Math.round(x * 100)}%`;
export const pct1 = (x) => `${(x * 100).toFixed(1).replace(".", ",")}%`;

export function plural(n, [one, few, many]) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

export function StatusDot({ status }) {
  return <i className={`dot dot--${status}`} aria-hidden="true" />;
}

function Pill({ ok, label, value, title }) {
  return (
    <span className={`pill ${ok ? "pill--ok" : "pill--off"}`} title={title || ""}>
      <i className="pill__dot" aria-hidden="true" />
      <span className="pill__label">{label}</span>
      <span className="pill__value">{value}</span>
    </span>
  );
}

export function Header({ status }) {
  const m = status?.model;
  return (
    <header className="topbar container">
      <a className="wordmark" href="#/" aria-label="На главную">
        <span className="display">Радар</span>
        <span className="wordmark__sub">слабых сигналов</span>
      </a>
      {status && (
        <div className="topbar__status">
          <Pill ok={m?.loaded} label="Модель" value={m?.cv_accuracy ? `точность ${pct1(m.cv_accuracy)}` : "не обучена"} />
          <Pill ok={status.llm?.available} label="LLM" value={status.llm?.available ? status.llm.model : "офлайн"}
                title={status.llm?.note} />
          <Pill ok={status.db} label="База" value={status.db ? "PostgreSQL" : "отключена"} />
        </div>
      )}
    </header>
  );
}

// Размер заголовка-запроса подбирается по самому длинному слову, чтобы слова не рвались
function titleSize(q) {
  const longest = Math.max(...q.split(/\s+/).map((w) => w.length), 4);
  const k = q.length > 48 ? 0.8 : 1;
  return { fontSize: `min(${Math.floor((k * 1080) / (longest * 0.46))}px, ${((k * 82) / (longest * 0.46)).toFixed(1)}vw, 150px)` };
}

export function Hero({ query, translation, loading, onSearch }) {
  const [value, setValue] = useState(query || "");
  useEffect(() => setValue(query || ""), [query]);
  const submit = (q) => q.trim().length > 1 && !loading && onSearch(q.trim());

  return (
    <section className="hero">
      <div className="hero__top">
        <p className="eyebrow">Газпромбанк.Тех · поиск зарождающихся технологий</p>
        <p className="hero__sources">arXiv · OpenAlex · веб RU · новости EN</p>
      </div>
      <h1 className={`display hero__title ${query ? "hero__title--query" : ""}`} style={query ? titleSize(query) : undefined}>
        {query || "Найти тренд до того, как о нём заговорят"}
      </h1>
      <form className="search" onSubmit={(e) => { e.preventDefault(); submit(value); }}>
        <input value={value} onChange={(e) => setValue(e.target.value)} aria-label="Технологическое направление"
               placeholder="Технологическое направление в свободной форме" />
        <button type="submit" disabled={loading}>{loading ? "Ищем…" : "Найти сигналы"}</button>
      </form>
      <div className="hero__foot">
        <div className="examples">
          {EXAMPLES.map((x) => (
            <button key={x} type="button" disabled={loading} onClick={() => { setValue(x); submit(x); }}>{x}</button>
          ))}
        </div>
        {translation && (
          <p className="hero__note">
            Поиск по науке: <b>{translation.en}</b> · {translation.method}
          </p>
        )}
      </div>
    </section>
  );
}

const STEPS = [
  ["Понимаем запрос", "Выделяем предметную область и переводим её для научных баз."],
  ["Собираем источники", "Препринты arXiv, статьи OpenAlex, русский веб и новости о стартапах."],
  ["Оцениваем", "Правила и ML-модель на 16 признаках, выведенных из критериев ТЗ."],
  ["Отсекаем лишнее", "Мейнстрим, хайп и шум исключаются — с причиной для каждого."],
];

export function HowItWorks() {
  return (
    <section className="section how">
      <h2 className="display section__title">Как работает радар</h2>
      <ol className="how__grid">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="how__step">
            <span className="display how__num">{pad(i + 1)}</span>
            <h3>{t}</h3>
            <p>{d}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

const LOADING_STEPS = ["Переводим запрос", "Собираем arXiv, OpenAlex и веб", "Группируем в технологии",
                       "Считаем признаки и уверенность", "Фильтруем мейнстрим и шум"];

export function Loading() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(s + 1, LOADING_STEPS.length - 1)), 1400);
    return () => clearInterval(t);
  }, []);
  return (
    <section className="section loading" aria-live="polite">
      <ol className="loading__steps">
        {LOADING_STEPS.map((s, i) => (
          <li key={s} className={i < step ? "is-done" : i === step ? "is-active" : ""}>{s}</li>
        ))}
      </ol>
      <div className="list list--skeleton">
        {Array.from({ length: 5 }, (_, i) => <div key={i} className="skeleton-row" />)}
      </div>
    </section>
  );
}

export function Stats({ result }) {
  const s = result.stats;
  const sources = Object.entries(s.by_source || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
  const cards = [
    { label: "Обработано источников", value: s.sources, sub: sources },
    { label: "Технологий-кандидатов", value: s.candidates, sub: `отсеяно фильтром: ${s.rejected}` },
    { label: "Слабых сигналов", value: s.signals,
      sub: s.low_trust ? `с пониженной достоверностью: ${s.low_trust}` : "прошли фильтр мейнстрима" },
    { label: "Уверенность выше 75%", value: s.confident, sub: "подтверждены правилами и моделью", accent: true },
  ];
  return (
    <section className="stats" aria-label="Статистика поиска">
      {cards.map((c) => (
        <div key={c.label} className={`stat ${c.accent ? "stat--accent" : ""}`}>
          <span className="stat__label">{c.label}</span>
          <span className="display stat__value">{c.value}</span>
          <span className="stat__sub">{c.sub}</span>
        </div>
      ))}
    </section>
  );
}

function Row({ card, href }) {
  const rejected = !["weak", "low_trust"].includes(card.status);
  return (
    <a className={`row ${rejected ? "row--rejected" : ""}`} href={href}>
      <span className="display row__rank">{card.rank ? pad(card.rank) : "—"}</span>
      <span className="row__main">
        <span className="row__name">{card.name}</span>
        <span className="row__meta">
          <StatusDot status={card.status} />
          {rejected ? card.status_reason : `${card.status_ru} · ${card.n_sources} ${plural(card.n_sources, ["источник", "источника", "источников"])}`}
        </span>
      </span>
      <span className="row__chips">
        {card.chips.map((c) => <span key={c} className="chip">{c}</span>)}
      </span>
      <span className="row__score">
        <span className="display row__pct">{card.score_pct}<small>%</small></span>
        <span className="bar"><i style={{ width: `${card.score_pct}%` }} /></span>
      </span>
      <span className="row__arrow" aria-hidden="true">→</span>
    </a>
  );
}

export function Results({ result }) {
  const [tab, setTab] = useState("signals");
  const list = tab === "signals" ? result.signals : result.rejected;
  const offset = tab === "signals" ? 0 : result.signals.length;
  return (
    <section className="section">
      <div className="section__head">
        <div>
          <h2 className="display section__title">
            {tab === "signals" ? `Топ-${result.signals.length} слабых сигналов` : "Отсеяно фильтром"}
          </h2>
          <p className="section__lead">
            {tab === "signals"
              ? "Ранние технологии, прошедшие фильтр мейнстрима, хайпа и шума. Откройте строку — там инсайт с обоснованием."
              : "Кандидаты, которые радар исключил: зрелые технологии, хайп и информационный шум — с причиной."}
          </p>
        </div>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "signals"} onClick={() => setTab("signals")}>
            Сигналы <span>{result.signals.length}</span>
          </button>
          <button role="tab" aria-selected={tab === "rejected"} onClick={() => setTab("rejected")}>
            Отсеяно <span>{result.rejected.length}</span>
          </button>
        </div>
      </div>
      {list.length ? (
        <div className="list">
          <div className="list__head" aria-hidden="true">
            <span>№</span><span>Технология</span><span>Ключевые предикторы</span><span>Уверенность</span><span />
          </div>
          {list.map((c, i) => <Row key={cardKey(c, offset + i)} card={c} href={`#/signal/${cardKey(c, offset + i)}`} />)}
        </div>
      ) : (
        <p className="empty">Здесь пусто — попробуйте сформулировать направление шире.</p>
      )}
      <p className="list__foot">
        Тексты: {result.llm?.available ? result.llm.model : "без LLM — выдержки из источников"}
        {result.llm?.note ? ` · ${result.llm.note}` : ""}
        {result.elapsed_sec != null ? ` · поиск занял ${String(result.elapsed_sec).replace(".", ",")} с` : ""}
      </p>
    </section>
  );
}

export function History({ items, currentId }) {
  if (!items?.length) return null;
  return (
    <section className="section history">
      <h2 className="display section__title section__title--small">Недавние запросы</h2>
      <ul className="history__list">
        {items.map((h) => (
          <li key={h.id}>
            <a href={`#/search/${h.id}`} className={h.id === currentId ? "is-current" : ""}>
              <span className="history__query">{h.query}</span>
              <span className="history__meta">
                {h.n_signals} {plural(h.n_signals, ["сигнал", "сигнала", "сигналов"])} · {h.n_sources} источн. ·{" "}
                {new Date(h.created_at).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
              </span>
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function Footer({ status }) {
  const m = status?.model;
  return (
    <footer className="footer container">
      <span className="display">Радар слабых сигналов</span>
      <span>
        Газпромбанк.Тех 2026 · {m?.name || "модель"}
        {m?.cv_accuracy ? ` · accuracy ${pct1(m.cv_accuracy)} на кросс-валидации` : ""}
      </span>
    </footer>
  );
}
