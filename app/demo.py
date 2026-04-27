"""Minimal browser demo for the recommendation API."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    """Serve a small UI that calls /recommend from the same FastAPI service."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Recipe Recommender Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f5ef;
      --ink: #17201b;
      --muted: #5e6a61;
      --line: #d8d2c4;
      --accent: #176b4d;
      --accent-dark: #0f4d37;
      --card: #fffefa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: min(980px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 24px;
    }
    h1 { font-size: 30px; line-height: 1.1; margin: 0 0 8px; font-weight: 750; }
    p { margin: 0; color: var(--muted); line-height: 1.5; }
    form {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) minmax(120px, 160px) auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 24px;
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; font-weight: 650; }
    input {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      padding: 9px 11px;
      font-size: 15px;
    }
    button {
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 0 16px;
      color: white;
      background: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    #status { min-height: 22px; margin-bottom: 14px; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
    }
    article {
      min-height: 128px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--card);
      padding: 14px;
      display: grid;
      align-content: space-between;
      gap: 14px;
    }
    h2 { margin: 0; font-size: 16px; line-height: 1.3; font-weight: 720; }
    .meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 640px) {
      header { display: block; }
      form { grid-template-columns: 1fr; }
      main { padding-top: 22px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Recipe Recommender</h1>
        <p>Generate top-N Food.com recommendations from the deployed model.</p>
      </div>
      <p><a href="/docs">API docs</a></p>
    </header>

    <form id="recommend-form">
      <label>User ID
        <input id="user-id" name="user_id" type="number" min="0" step="1" value="123" required>
      </label>
      <label>Count
        <input id="top-n" name="top_n" type="number" min="1" max="20" step="1" value="10" required>
      </label>
      <button type="submit">Recommend</button>
    </form>

    <div id="status"></div>
    <section id="results" class="grid" aria-live="polite"></section>
  </main>

  <script>
    const form = document.querySelector("#recommend-form");
    const statusEl = document.querySelector("#status");
    const resultsEl = document.querySelector("#results");

    function renderRecommendations(items, model) {
      resultsEl.innerHTML = "";
      for (const item of items) {
        const card = document.createElement("article");
        const title = document.createElement("h2");
        title.textContent = item.name;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `<span>ID ${item.recipe_id}</span><strong>${item.predicted_rating.toFixed(2)}</strong>`;
        card.append(title, meta);
        resultsEl.append(card);
      }
      statusEl.textContent = `${items.length} recommendations from ${model}`;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.textContent = "Loading recommendations...";
      resultsEl.innerHTML = "";
      const payload = {
        user_id: Number(document.querySelector("#user-id").value),
        top_n: Number(document.querySelector("#top-n").value),
        exclude_rated: true
      };
      try {
        const response = await fetch("/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const body = await response.json();
        if (!response.ok) {
          throw new Error(body.detail || `HTTP ${response.status}`);
        }
        renderRecommendations(body.recommendations, body.model);
      } catch (error) {
        statusEl.textContent = `Request failed: ${error.message}`;
      }
    });
  </script>
</body>
</html>"""
