const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Ошибка сервера (${res.status})`);
  }
  return res.json();
}

export const api = {
  status: () => request("/status"),
  history: () => request("/history"),
  search: (query) => request("/search", { method: "POST", body: JSON.stringify({ query, top_k: 15 }) }),
  getSearch: (id) => request(`/searches/${id}`),
  getSignal: (id) => request(`/signals/${id}`),
};
