import { useEffect, useState } from "react";
import { api } from "./api.js";
import { cardKey, Footer, Header, Hero, History, HowItWorks, Loading, Results, Stats } from "./components.jsx";
import Insight from "./Insight.jsx";

function parseHash() {
  const [, page, id] = (window.location.hash.replace(/^#/, "") || "/").split("/");
  if ((page === "search" || page === "signal") && id) return { page, id };
  return { page: "home" };
}

export default function App() {
  const [route, setRoute] = useState(parseHash);
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [signal, setSignal] = useState({ card: null, loading: false, error: "" });

  const refreshHistory = () => api.history().then(setHistory).catch(() => {});

  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    api.status().then(setStatus).catch(() => {});
    refreshHistory();
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Прошлый запрос по ссылке #/search/<id> — из PostgreSQL
  useEffect(() => {
    if (route.page !== "search" || String(result?.search_id) === route.id) return;
    setLoading(true);
    setError("");
    api.getSearch(route.id).then(setResult).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }, [route]);

  // Страница инсайта: сначала ищем карточку в текущей выдаче, иначе берём из базы
  useEffect(() => {
    if (route.page !== "signal") return;
    window.scrollTo(0, 0);
    const all = result ? [...result.signals, ...result.rejected] : [];
    const local = all.find((c, i) => cardKey(c, i) === route.id);
    if (local) return setSignal({ card: local, loading: false, error: "" });
    setSignal({ card: null, loading: true, error: "" });
    api.getSignal(route.id)
      .then((card) => setSignal({ card, loading: false, error: "" }))
      .catch((e) => setSignal({ card: null, loading: false, error: e.message }));
  }, [route, result]);

  const onSearch = (query) => {
    setLoading(true);
    setError("");
    setResult(null);
    api.search(query)
      .then((res) => {
        setResult(res);
        refreshHistory();
        api.status().then(setStatus).catch(() => {});
        window.location.hash = res.search_id ? `#/search/${res.search_id}` : "#/";
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const back = () => {
    window.location.hash = result?.search_id ? `#/search/${result.search_id}` : "#/";
  };

  if (route.page === "signal") {
    return (
      <>
        <Header status={status} />
        <Insight {...signal} query={result?.query} onBack={back} />
        <Footer status={status} />
      </>
    );
  }

  return (
    <>
      <Header status={status} />
      <main className="container">
        <Hero query={result?.query} translation={result?.translation} loading={loading} onSearch={onSearch} />
        {loading && <Loading />}
        {!loading && error && (
          <div className="notice" role="alert">
            <span className="display">Не получилось</span>
            <p>{error}. Проверьте, что API запущен, и повторите запрос.</p>
          </div>
        )}
        {!loading && result && (
          <>
            <Stats result={result} />
            <Results result={result} />
          </>
        )}
        {!loading && !result && !error && <HowItWorks />}
        <History items={history} currentId={result?.search_id} />
      </main>
      <Footer status={status} />
    </>
  );
}
