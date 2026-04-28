const { useState, useEffect, useCallback, useRef } = React;

// ── Food particle burst ────────────────────────────────────────────────────
const FOOD_EMOJIS = ['🍳','🥘','🥗','🍜','🧄','🌿','🫙','🥩','🍋','🧅','🫚','🌶️'];

function Particles({ trigger }) {
  const [particles, setParticles] = useState([]);
  const prevTrigger = useRef(null);

  useEffect(() => {
    if (!trigger || trigger === prevTrigger.current) return;
    prevTrigger.current = trigger;
    const ps = Array.from({ length: 10 }, (_, i) => ({
      id: Date.now() + i,
      emoji: FOOD_EMOJIS[Math.floor(Math.random() * FOOD_EMOJIS.length)],
      left: 8 + Math.random() * 84,
      delay: Math.random() * 0.5,
      size: 15 + Math.random() * 10,
      duration: 1.6 + Math.random() * 0.6,
    }));
    setParticles(ps);
    const t = setTimeout(() => setParticles([]), 2400);
    return () => clearTimeout(t);
  }, [trigger]);

  if (!particles.length) return null;
  return (
    <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 200, overflow: 'hidden' }}>
      {particles.map(p => (
        <span key={p.id} className="particle" style={{
          left: `${p.left}%`, fontSize: p.size,
          animationDelay: `${p.delay}s`, animationDuration: `${p.duration}s`,
        }}>{p.emoji}</span>
      ))}
    </div>
  );
}

// ── Explanation box with typewriter effect ─────────────────────────────────
function ExplBox({ text, source, loading }) {
  const [shown, setShown] = useState('');
  useEffect(() => {
    if (!text) { setShown(''); return; }
    setShown('');
    let i = 0;
    const iv = setInterval(() => {
      i++;
      setShown(text.slice(0, i));
      if (i >= text.length) clearInterval(iv);
    }, 12);
    return () => clearInterval(iv);
  }, [text]);

  return (
    <div className="expl-box">
      <div className="expl-header">
        <span className="expl-label">Why this?</span>
        {source && <Badge label={source} variant={source === 'LLM' ? 'gold' : source === 'fallback' ? 'neutral' : 'gold'} />}
        {loading && <span className="spinner" style={{ width: 11, height: 11, borderWidth: 1.5 }}></span>}
      </div>
      {loading && !shown && <span style={{ color: 'var(--text-3)', fontSize: 11 }}>Generating explanation…</span>}
      {shown && <p style={{ margin: 0 }}>{shown}</p>}
    </div>
  );
}

// ── Recipe card ────────────────────────────────────────────────────────────
function RecipeCard({ recipe, type, index, context, onFindSimilar }) {
  const [explaining, setExplaining] = useState(false);
  const [expl, setExpl]             = useState(null);
  const [explSrc, setExplSrc]       = useState(null);

  const isMF     = type === 'existing';
  const score    = isMF ? recipe.predicted_rating : recipe.score;
  const scoreMax = isMF ? 5 : 1;
  const scoreLbl = isMF ? 'pred. rating' : 'similarity';
  const srcClass = isMF ? 's-mf' : type === 'similar' ? 's-faiss' : 's-cold';

  async function handleExplain() {
    if (expl !== null) { setExpl(null); setExplSrc(null); return; }
    setExplaining(true);

    const rec = isMF
      ? { recipe_id: recipe.recipe_id, name: recipe.name, predicted_rating: recipe.predicted_rating }
      : { recipe_id: recipe.recipe_id, name: recipe.name, score: recipe.score || 0 };

    const payload = isMF
      ? { user_id: context.userId || 0, recommendations: [rec], top_n: 1 }
      : { liked_recipe_ids: context.likedIds || [], disliked_recipe_ids: context.dislikedIds || [], recommendations: [rec], top_n: 1 };

    const backendResult = await api.explain(payload);

    if (backendResult?.explanation) {
      setExpl(backendResult.explanation);
      setExplSrc(backendResult.source || 'LLM');
    } else {
      setExpl(`"${recipe.name}" was selected because it ranks highly for the current recommendation context. The explanation service was unavailable, so this local fallback is shown.`);
      setExplSrc('fallback');
    }
    setExplaining(false);
  }

  return (
    <div className={`recipe-card ${srcClass}`} style={{ animationDelay: `${index * 0.048}s` }}>
      <div className="card-top">
        <span className="card-name">{recipe.name}</span>
        <div className="card-score-wrap">
          <span className="card-score-val">{score?.toFixed(2)}</span>
          <span className="card-score-lbl">{scoreLbl}</span>
        </div>
      </div>

      <ScoreBar value={score} max={scoreMax} color={isMF ? 'ember' : 'teal'} delay={index * 50} />

      <div className="card-meta">
        <span className="card-id">#{recipe.recipe_id}</span>
        {isMF && <Badge label="TimeAwareMF" variant="ember" />}
        {(type === 'similar' || type === 'newuser') && <Badge label={recipe.backend || 'FAISS'} variant="teal" />}
        {type === 'newuser' && <Badge label="Cold-start" variant="neutral" />}
      </div>

      <div className="card-actions">
        <button className="btn-sm why" onClick={handleExplain}>
          {expl !== null ? '✕ close' : '✦ Why this?'}
        </button>
        {isMF && onFindSimilar && (
          <button className="btn-sm similar" onClick={() => onFindSimilar(recipe.recipe_id)}>
            Find similar →
          </button>
        )}
      </div>

      {(explaining || expl !== null) && (
        <ExplBox text={expl} source={explSrc} loading={explaining} />
      )}
    </div>
  );
}

// ── Existing User Tab ──────────────────────────────────────────────────────
function ExistingUserTab({ onFindSimilar }) {
  const [userId,       setUserId]       = useState('');
  const [topN,         setTopN]         = useState(10);
  const [excludeRated, setExcludeRated] = useState(true);
  const [loading,      setLoading]      = useState(false);
  const [results,      setResults]      = useState(null);
  const [error,        setError]        = useState(null);
  const [burst,        setBurst]        = useState(null);

  async function run(uid) {
    const id = uid ?? userId;
    if (!id) return;
    setLoading(true); setError(null);
    try {
      const data = await api.recommend(id, topN, excludeRated);
      setResults(data);
      setBurst(Date.now());
    }
    catch (e) { setError(e.message); }
    setLoading(false);
  }

  const ctx = { userId: parseInt(userId) || 0 };

  return (
    <div className="tab-pane">
      <Particles trigger={burst} />
      <div className="panel-left">
        <div className="ctrl-section">
          <label className="ctrl-label">User ID</label>
          <input className="ctrl-input ctrl-input-mono" placeholder="e.g. 123456"
            value={userId} onChange={e => setUserId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && run()} />
          <div style={{ marginTop: 3 }}>
            <div className="quick-try">
              <span className="ctrl-hint">Quick try:</span>
              {[80901, 226863, 424680].map(id => (
                <button key={id} className="quick-id" onClick={() => { setUserId(String(id)); run(String(id)); }}>
                  #{id}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="ctrl-section">
          <div className="ctrl-row">
            <label className="ctrl-label">Top N</label>
            <Stepper value={topN} onChange={setTopN} min={1} max={20} />
          </div>
        </div>

        <div className="ctrl-section">
          <div className="toggle-wrap">
            <span className="toggle-label">Exclude rated</span>
            <Toggle value={excludeRated} onChange={setExcludeRated} />
          </div>
        </div>

        <div className="ctrl-sep"></div>
        <button className="btn-primary" onClick={() => run()} disabled={!userId || loading}>
          {loading
            ? <span style={{ display: 'flex', alignItems: 'center', gap: 7, justifyContent: 'center' }}>
                <span className="spinner" style={{ borderTopColor: '#0f0f0e' }}></span> Loading…
              </span>
            : 'Get recommendations'}
        </button>

        <div className="ctrl-footer">
          Collaborative filtering via <span style={{ color: 'var(--ember)' }}>TimeAwareMF</span>.
          Predicts ratings from your interaction history on the Food.com dataset.
        </div>
      </div>

      <div className="panel-right">
        {!results && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">👤</div>
            <div className="empty-title">Enter a user ID to begin</div>
            <div className="empty-sub">Collaborative filtering generates personalized recommendations from your rating history.</div>
          </div>
        )}
        {error && <div className="err-box">Error: {error}</div>}
        {loading && Array.from({ length: Math.min(topN, 6) }).map((_, i) => <SkeletonCard key={i} index={i} />)}
        {results && !loading && (
          <>
            <div className="results-hdr">
              <span className="results-title">Recommendations for user #{userId}</span>
              <span className="results-count">{results.length} results</span>
            </div>
            {results.map((r, i) => (
              <RecipeCard key={r.recipe_id} recipe={r} type="existing" index={i} context={ctx} onFindSimilar={onFindSimilar} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ── New User Tab ───────────────────────────────────────────────────────────
function NewUserTab({ onFindSimilar }) {
  const [likedInput,    setLikedInput]    = useState('');
  const [dislikedInput, setDislikedInput] = useState('');
  const [chips,         setChips]         = useState([]);
  const [maxMin,        setMaxMin]        = useState(60);
  const [topN,          setTopN]          = useState(10);
  const [loading,       setLoading]       = useState(false);
  const [results,       setResults]       = useState(null);
  const [error,         setError]         = useState(null);
  const [burst,         setBurst]         = useState(null);
  const [poppingChip,   setPoppingChip]   = useState(null);

  const allChips = ['chicken', 'pasta', 'rice', 'vegetarian', 'quick', 'dessert', 'spicy', 'seafood'];
  const parseIds = str => str.split(/[\s,]+/).map(s => parseInt(s)).filter(n => !isNaN(n));

  function toggleChip(c) {
    setChips(prev => {
      const wasActive = prev.includes(c);
      if (!wasActive) {
        setPoppingChip(c);
        setTimeout(() => setPoppingChip(null), 320);
      }
      return wasActive ? prev.filter(x => x !== c) : [...prev, c];
    });
  }

  async function run() {
    setLoading(true); setError(null);
    const likedIds    = parseIds(likedInput);
    const dislikedIds = parseIds(dislikedInput);
    const prefs       = { ingredients: chips, avoid: [], max_minutes: maxMin };
    try {
      const data = await api.recommendNewUser(likedIds, dislikedIds, prefs, topN);
      setResults(data);
      setBurst(Date.now());
    }
    catch (e) { setError(e.message); }
    setLoading(false);
  }

  const ctx = { likedIds: parseIds(likedInput), dislikedIds: parseIds(dislikedInput) };

  return (
    <div className="tab-pane">
      <Particles trigger={burst} />
      <div className="panel-left">
        <div className="ctrl-section">
          <label className="ctrl-label">Liked Recipe IDs</label>
          <textarea className="ctrl-input ctrl-input-mono" rows={2}
            placeholder="e.g. 456, 789, 24576" value={likedInput}
            onChange={e => setLikedInput(e.target.value)} />
          <div className="quick-try">
            <span className="ctrl-hint">Quick try:</span>
            {[24576, 8701, 15302].map(id => (
              <button key={id} className="quick-id"
                onClick={() => setLikedInput(p => p ? `${p}, ${id}` : String(id))}>+{id}</button>
            ))}
          </div>
        </div>

        <div className="ctrl-section">
          <label className="ctrl-label">Disliked Recipe IDs</label>
          <textarea className="ctrl-input ctrl-input-mono" rows={2}
            placeholder="e.g. 321, 654" value={dislikedInput}
            onChange={e => setDislikedInput(e.target.value)} />
        </div>

        <div className="ctrl-section">
          <label className="ctrl-label">Preferences</label>
          <div className="chips">
            {allChips.map(c => (
              <span key={c}
                className={`chip${chips.includes(c) ? ' active' : ''}${poppingChip === c ? ' popping' : ''}`}
                onClick={() => toggleChip(c)}>{c}</span>
            ))}
          </div>
        </div>

        <div className="ctrl-section">
          <div className="ctrl-row">
            <label className="ctrl-label">Max minutes</label>
            <Stepper value={maxMin} onChange={setMaxMin} min={15} max={240} />
          </div>
        </div>

        <div className="ctrl-section">
          <div className="ctrl-row">
            <label className="ctrl-label">Top N</label>
            <Stepper value={topN} onChange={setTopN} min={1} max={20} />
          </div>
        </div>

        <div className="ctrl-sep"></div>
        <button className="btn-primary" onClick={run} disabled={loading}>
          {loading
            ? <span style={{ display: 'flex', alignItems: 'center', gap: 7, justifyContent: 'center' }}>
                <span className="spinner" style={{ borderTopColor: '#0f0f0e' }}></span> Building profile…
              </span>
            : 'Recommend for me'}
        </button>

        <div className="ctrl-footer">
          Cold-start via <span style={{ color: 'var(--teal)' }}>semantic FAISS</span> search.
          Builds a taste profile from liked recipe embeddings—no user history required.
        </div>
      </div>

      <div className="panel-right">
        {!results && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">✨</div>
            <div className="empty-title">No user ID? No problem.</div>
            <div className="empty-sub">Add recipe IDs you enjoy or select preference chips. We'll build a semantic profile on the fly.</div>
          </div>
        )}
        {error && <div className="err-box">Error: {error}</div>}
        {loading && Array.from({ length: Math.min(topN, 6) }).map((_, i) => <SkeletonCard key={i} index={i} />)}
        {results && !loading && (
          <>
            <div className="results-hdr">
              <span className="results-title">Cold-start recommendations</span>
              <span className="results-count">{results.length} results</span>
            </div>
            {results.map((r, i) => (
              <RecipeCard key={r.recipe_id} recipe={r} type="newuser" index={i} context={ctx} onFindSimilar={onFindSimilar} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ── Similar Recipes Tab ────────────────────────────────────────────────────
function SimilarTab({ initialRecipeId }) {
  const [recipeId, setRecipeId] = useState(initialRecipeId ? String(initialRecipeId) : '');
  const [topN,     setTopN]     = useState(10);
  const [loading,  setLoading]  = useState(false);
  const [results,  setResults]  = useState(null);
  const [error,    setError]    = useState(null);
  const [burst,    setBurst]    = useState(null);

  async function run(rid) {
    const id = rid ?? recipeId;
    if (!id) return;
    setLoading(true); setError(null);
    try {
      const data = await api.similar(id, topN);
      setResults(data);
      setBurst(Date.now());
    }
    catch (e) { setError(e.message); }
    setLoading(false);
  }

  useEffect(() => {
    if (initialRecipeId) { setRecipeId(String(initialRecipeId)); run(String(initialRecipeId)); }
  }, [initialRecipeId]);

  return (
    <div className="tab-pane">
      <Particles trigger={burst} />
      <div className="panel-left">
        <div className="ctrl-section">
          <label className="ctrl-label">Recipe ID</label>
          <input className="ctrl-input ctrl-input-mono" placeholder="e.g. 24576"
            value={recipeId} onChange={e => setRecipeId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && run()} />
          <div className="quick-try">
            <span className="ctrl-hint">Quick try:</span>
            {[24576, 153501, 44203].map(id => (
              <button key={id} className="quick-id" onClick={() => { setRecipeId(String(id)); run(String(id)); }}>
                #{id}
              </button>
            ))}
          </div>
        </div>

        <div className="ctrl-section">
          <div className="ctrl-row">
            <label className="ctrl-label">Top N</label>
            <Stepper value={topN} onChange={setTopN} min={1} max={20} />
          </div>
        </div>

        <div className="ctrl-sep"></div>
        <button className="btn-primary" onClick={() => run()} disabled={!recipeId || loading}>
          {loading
            ? <span style={{ display: 'flex', alignItems: 'center', gap: 7, justifyContent: 'center' }}>
                <span className="spinner" style={{ borderTopColor: '#0f0f0e' }}></span> Searching…
              </span>
            : 'Find similar'}
        </button>

        <div className="ctrl-footer">
          Recipe-to-recipe nearest-neighbor search via <span style={{ color: 'var(--teal)' }}>FAISS</span> index
          on sentence-transformer recipe embeddings.
        </div>
      </div>

      <div className="panel-right">
        {!results && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <div className="empty-title">Enter a recipe ID</div>
            <div className="empty-sub">FAISS nearest-neighbor search retrieves recipes with the highest embedding similarity.</div>
          </div>
        )}
        {error && <div className="err-box">Error: {error}</div>}
        {loading && Array.from({ length: Math.min(topN, 6) }).map((_, i) => <SkeletonCard key={i} index={i} />)}
        {results && !loading && (
          <>
            <div className="results-hdr">
              <span className="results-title">Similar to recipe #{recipeId}</span>
              <span className="results-count">{results.length} results</span>
            </div>
            {results.map((r, i) => (
              <RecipeCard key={r.recipe_id} recipe={r} type="similar" index={i} context={{}} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ── Metrics Tab ────────────────────────────────────────────────────────────
function MetricsTab() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  async function load() {
    setLoading(true); setError(null);
    try { setData(await api.metrics()); }
    catch (e) { setError(e.message); }
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  const totalReqs = data ? data.latency.reduce((a, r) => a + r.count, 0) : 0;
  const avgLat    = data ? Math.round(data.latency.reduce((a, r) => a + r.avg_ms, 0) / data.latency.length) : 0;

  return (
    <div className="panel-full">
      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-3)' }}>
          <span className="spinner"></span> Loading metrics…
        </div>
      )}
      {error && <div className="err-box">Error: {error}</div>}
      {data && !loading && (
        <div style={{ maxWidth: 760 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
            <Badge label={data.model} variant="ember" />
            <Badge label={data.vector_backend} variant="teal" />
            <Badge label="LLM explanation" variant="gold" />
            <button className="btn-icon" onClick={load} style={{ marginLeft: 'auto' }}>↻ Refresh</button>
          </div>

          <div className="metrics-stat-grid">
            <div className="m-stat">
              <div className="m-stat-lbl">Total requests</div>
              <div className="m-stat-val"><AnimatedNum value={totalReqs} /></div>
            </div>
            <div className="m-stat">
              <div className="m-stat-lbl">Mean latency</div>
              <div className="m-stat-val" style={{ color: 'var(--gold)' }}>
                <AnimatedNum value={avgLat} /><span className="m-stat-unit">ms</span>
              </div>
            </div>
            <div className="m-stat">
              <div className="m-stat-lbl">Routes tracked</div>
              <div className="m-stat-val"><AnimatedNum value={data.latency.length} /></div>
            </div>
          </div>

          <div className="m-table-wrap">
            <div className="m-table-header">
              <span className="m-table-title">Route Latency</span>
              <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'monospace' }}>milliseconds</span>
            </div>
            <table className="m-table">
              <thead>
                <tr>
                  <th>Route</th>
                  <th className="r">Count</th>
                  <th className="r">Avg</th>
                  <th className="r">p95</th>
                  <th className="r">Max</th>
                </tr>
              </thead>
              <tbody>
                {data.latency.map(row => (
                  <tr key={row.route}>
                    <td className="route-cell">{row.route}</td>
                    <td className="mono r"><AnimatedNum value={row.count} /></td>
                    <td className="mono r" style={{ color: row.avg_ms > 500 ? 'var(--ember)' : 'var(--text)' }}>
                      {row.avg_ms}
                    </td>
                    <td className="mono r" style={{ color: row.p95_ms > 1000 ? 'var(--ember)' : 'var(--text-2)' }}>
                      {row.p95_ms}
                    </td>
                    <td className="mono r" style={{ color: 'var(--text-3)' }}>{row.max_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { ExplBox, RecipeCard, ExistingUserTab, NewUserTab, SimilarTab, MetricsTab });
