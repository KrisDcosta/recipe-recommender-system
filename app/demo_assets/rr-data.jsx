// ── API layer ──────────────────────────────────────────────────────────────
const api = {
  get baseURL() { return window._rrApiBase || ''; },

  async health() {
    try {
      const res = await fetch(`${this.baseURL}/health`, { signal: AbortSignal.timeout(3000) });
      return res.ok ? 'healthy' : 'error';
    } catch { return 'unknown'; }
  },

  async recommend(userId, topN, excludeRated) {
    try {
      const res = await fetch(`${this.baseURL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: parseInt(userId), top_n: topN, exclude_rated: excludeRated }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return data.recommendations || data;
    } catch (error) { throw error; }
  },

  async recommendNewUser(likedIds, dislikedIds, prefs, topN) {
    try {
      const res = await fetch(`${this.baseURL}/recommend/new-user`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ liked_recipe_ids: likedIds, disliked_recipe_ids: dislikedIds, preferences: prefs, top_n: topN }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return (data.recommendations || []).map(item => ({
        ...item,
        backend: data.search_backend || item.backend || 'semantic',
      }));
    } catch (error) { throw error; }
  },

  async similar(recipeId, topN) {
    try {
      const res = await fetch(`${this.baseURL}/similar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ recipe_id: parseInt(recipeId), top_n: topN }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return (data.similar || []).map(item => ({
        recipe_id: item.recipe_id,
        name: item.name,
        score: item.similarity,
        backend: data.search_backend || 'semantic',
      }));
    } catch (error) { throw error; }
  },

  async explain(payload) {
    try {
      const res = await fetch(`${this.baseURL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const first = data.explanations?.[0];
      return first ? {
        explanation: first.explanation,
        source: data.fallback ? 'fallback' : 'LLM',
      } : null;
    } catch { return null; }
  },

  async metrics() {
    try {
      const res = await fetch(`${this.baseURL}/metrics`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return {
        model: data.model,
        vector_backend: data.vector_store,
        latency: Object.entries(data.latency || {}).map(([route, vals]) => ({
          route,
          count: vals.count || 0,
          avg_ms: vals.avg_ms || 0,
          p95_ms: vals.p95_ms || 0,
          max_ms: vals.max_ms || 0,
        })),
      };
    } catch (error) { throw error; }
  },
};

Object.assign(window, { api });
