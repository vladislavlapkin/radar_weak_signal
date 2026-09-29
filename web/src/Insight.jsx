import { pad, pct, plural, StatusDot } from "./components.jsx";

function Section({ title, children }) {
  return (
    <section className="doc__section">
      <h2 className="doc__label">{title}</h2>
      {children}
    </section>
  );
}

// «Ранняя стадия: концепция, прототип» -> «концепция, прототип»; дубль названия не показываем
const detail = (p) => {
  const d = p.detail || "";
  if (!d.toLowerCase().startsWith(p.label.toLowerCase())) return d;
  const rest = d.slice(p.label.length).replace(/^[:\s—-]+/, "");
  return rest ? rest[0].toUpperCase() + rest.slice(1) : "";
};

function Predictors({ items }) {
  const groups = [
    ["модель", "ML-модель · вклад признака в решение"],
    ["правила", "Правила · значение от 0 до 1"],
  ];
  return groups.map(([source, title]) => {
    const list = items.filter((p) => p.source === source);
    if (!list.length) return null;
    const max = Math.max(...list.map((p) => p.value), source === "правила" ? 1 : 0.01);
    return (
      <div key={source} className="predictors">
        <h3 className="predictors__title">{title}</h3>
        {list.map((p) => (
          <div key={p.label + p.detail} className={`predictor ${p.effect === "+" ? "is-pos" : "is-neg"}`}>
            <div className="predictor__text">
              <span className="predictor__label">{p.label}</span>
              {detail(p) && <span className="predictor__detail">{detail(p)}</span>}
            </div>
            <div className="predictor__bar" aria-hidden="true"><i style={{ width: `${(p.value / max) * 100}%` }} /></div>
            <span className="predictor__value">{p.effect}{p.value.toFixed(2).replace(".", ",")}</span>
          </div>
        ))}
      </div>
    );
  });
}

function Sources({ items }) {
  return (
    <ol className="sources">
      {items.map((s, i) => (
        <li key={s.url + i} className="source">
          <div className="source__meta">
            <span>{s.type_ru}</span>
            <span>{s.site}</span>
            {s.date && <span>{new Date(s.date).toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" })}</span>}
            <span>{s.lang}</span>
            <span className={`trust trust--${s.trust}`}>Доверие: {s.trust_ru.toLowerCase()}</span>
          </div>
          <a className="source__title" href={s.url} target="_blank" rel="noreferrer">{s.title} ↗</a>
          <p className="source__summary">{s.summary}</p>
          <p className="source__note">{s.summary_note}</p>
        </li>
      ))}
    </ol>
  );
}

export default function Insight({ card, loading, error, query: currentQuery, onBack }) {
  if (loading) return <main className="container insight"><div className="skeleton-row skeleton-row--tall" /></main>;
  if (error || !card) {
    return (
      <main className="container insight">
        <button className="back" onClick={onBack}>← К выдаче</button>
        <div className="notice"><span className="display">Инсайт не найден</span><p>{error || "Откройте его из выдачи."}</p></div>
      </main>
    );
  }
  const parts = card.score_parts;
  const query = card.query || currentQuery;
  const years = [...new Set(card.sources.map((s) => s.date?.slice(0, 4)).filter(Boolean))].sort();
  const types = [...new Set(card.sources.map((s) => s.type_ru))];
  const isSignal = ["weak", "low_trust"].includes(card.status);

  return (
    <main className="container insight">
      <button className="back" onClick={onBack}>← К выдаче</button>
      <header className="insight__head">
        <div className="insight__intro">
          <p className="eyebrow eyebrow--ink">
            Инсайт{card.rank ? ` · сигнал ${pad(card.rank)}` : ""}{query ? ` · «${query}»` : ""}
          </p>
          <h1 className="insight__title">{card.name}</h1>
          <div className="insight__tags">
            <span className={`status status--${card.status}`}><StatusDot status={card.status} />{card.status_ru}</span>
            {card.keyphrases?.map((k) => <span key={k} className="chip">{k}</span>)}
          </div>
        </div>
        <aside className="scorecard">
          <span className="scorecard__label">Уверенность модели</span>
          <span className="display scorecard__value">{card.score_pct}<small>%</small></span>
          <span className="scorecard__conf">{card.confidence_ru} уверенность</span>
          <dl className="scorecard__parts">
            <div><dt>Правила</dt><dd className="display">{pct(parts.rules)}</dd></div>
            <div><dt>ML-модель</dt><dd className="display">{parts.ml == null ? "—" : pct(parts.ml)}</dd></div>
          </dl>
          <span className="scorecard__formula">{parts.formula}</span>
        </aside>
      </header>

      <div className="insight__grid">
        <article className="doc">
          <Section title="Описание технологии">
            <p className="doc__lead">{card.description}</p>
            {card.quote && (
              <blockquote className="quote">
                <p>{card.quote.text}</p>
                <cite>Цитата из источника · {card.quote.site} · {card.quote.lang}</cite>
              </blockquote>
            )}
          </Section>
          <Section title="Потенциальное преимущество">
            <p>{card.advantage}</p>
            {card.advantage_source && <p className="doc__note">Формулировка из источника: {card.advantage_source}</p>}
          </Section>
          <Section title="Кейс-пример">
            <p>{card.case.text}</p>
            {card.case.url && <a className="link" href={card.case.url} target="_blank" rel="noreferrer">Открыть первоисточник ↗</a>}
          </Section>
          <Section title={isSignal ? "Почему это слабый сигнал" : "Почему кандидат исключён"}>
            <ul className="why">{card.why.map((w) => <li key={w}>{w}</li>)}</ul>
          </Section>
          <Section title="Ключевые предикторы">
            <Predictors items={card.predictors} />
          </Section>
          <Section title={`Источники · ${card.sources.length} из ${card.n_sources}`}>
            <Sources items={card.sources} />
          </Section>
        </article>

        <aside className="passport">
          <h2 className="doc__label">Паспорт сигнала</h2>
          <dl>
            <div><dt>Статус</dt><dd>{card.status_ru}</dd></div>
            <div><dt>Причина</dt><dd>{card.status_reason}</dd></div>
            <div><dt>Источников</dt><dd>{card.n_sources} {plural(card.n_sources, ["документ", "документа", "документов"])}</dd></div>
            <div><dt>Типы</dt><dd>{types.join(", ")}</dd></div>
            <div><dt>Период</dt><dd>{years.length ? (years.length > 1 ? `${years[0]}–${years.at(-1)}` : years[0]) : "без дат"}</dd></div>
            <div><dt>Тексты</dt><dd>{card.llm_model}</dd></div>
            <div><dt>Сформирован</dt><dd>{new Date(card.built_at).toLocaleDateString("ru-RU")}</dd></div>
          </dl>
          <button className="button-ghost" onClick={() => window.print()}>Сохранить как PDF</button>
        </aside>
      </div>
    </main>
  );
}
