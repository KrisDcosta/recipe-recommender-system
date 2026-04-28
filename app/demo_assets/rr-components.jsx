const { useState, useEffect, useRef } = React;

// ── Badge ──────────────────────────────────────────────────────────────────
function Badge({ label, variant = 'neutral' }) {
  return <span className={`badge badge-${variant}`}>{label}</span>;
}

// ── Score bar ──────────────────────────────────────────────────────────────
function ScoreBar({ value, max = 5, color = 'ember', delay = 0 }) {
  const [pct, setPct] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setPct(Math.round((value / max) * 100)), 60 + delay);
    return () => clearTimeout(t);
  }, [value, max, delay]);
  return (
    <div className="score-track">
      <div className={`score-fill ${color}`} style={{ width: `${pct}%` }}></div>
    </div>
  );
}

// ── Skeleton card ──────────────────────────────────────────────────────────
function SkeletonCard({ index }) {
  return (
    <div className="skel-card" style={{ animationDelay: `${index * 0.06}s` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
        <div className="skel" style={{ height: 15, width: '58%', borderRadius: 4 }}></div>
        <div className="skel" style={{ height: 15, width: 38 }}></div>
      </div>
      <div className="score-track"><div className="skel" style={{ height: 3, width: '100%', borderRadius: 2 }}></div></div>
      <div style={{ display: 'flex', gap: 5 }}>
        <div className="skel" style={{ height: 19, width: 70 }}></div>
        <div className="skel" style={{ height: 19, width: 52 }}></div>
      </div>
      <div style={{ display: 'flex', gap: 5 }}>
        <div className="skel" style={{ height: 22, width: 78 }}></div>
        <div className="skel" style={{ height: 22, width: 78 }}></div>
      </div>
    </div>
  );
}

// ── Animated number (count-up) ─────────────────────────────────────────────
function AnimatedNum({ value, duration = 900 }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let raf;
    const start = performance.now();
    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(Math.round(value * eased));
      if (p < 1) raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return <>{display.toLocaleString()}</>;
}

// ── Toggle ─────────────────────────────────────────────────────────────────
function Toggle({ value, onChange }) {
  return (
    <div className={`toggle ${value ? 'on' : ''}`} onClick={() => onChange(!value)} role="switch" aria-checked={value}>
      <div className="toggle-thumb"></div>
    </div>
  );
}

// ── Stepper ────────────────────────────────────────────────────────────────
function Stepper({ value, onChange, min = 1, max = 20 }) {
  return (
    <div className="stepper">
      <button className="stepper-btn" onClick={() => onChange(Math.max(min, value - 1))}>−</button>
      <span className="stepper-val">{value}</span>
      <button className="stepper-btn" onClick={() => onChange(Math.min(max, value + 1))}>+</button>
    </div>
  );
}

// ── Sun / Moon SVG icons ───────────────────────────────────────────────────
function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="5"/>
      <line x1="12" y1="2"  x2="12" y2="4"/>
      <line x1="12" y1="20" x2="12" y2="22"/>
      <line x1="4.22" y1="4.22"  x2="5.64" y2="5.64"/>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="2" y1="12" x2="4"  y2="12"/>
      <line x1="20" y1="12" x2="22" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  );
}
function MoonIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  );
}

// ── Header ─────────────────────────────────────────────────────────────────
function Header({ health, isDark, onThemeToggle }) {
  const base = window._rrApiBase || '';
  const healthLabel = health === 'healthy' ? 'API online' : health === 'error' ? 'API error' : 'checking…';
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <div className="header-logo">🍳</div>
          <span className="header-title">Recipe Recommender</span>
        </div>
        <div className="header-sep"></div>
        <div className="header-badges">
          <Badge label="TimeAwareMF" variant="ember" />
          <Badge label="FAISS" variant="teal" />
          <Badge label="LLM" variant="gold" />
        </div>
        <div className="header-right">
          <div className="health-wrap">
            <div className={`health-dot ${health}`}></div>
            <span className="health-text">{healthLabel}</span>
          </div>
          <a href={`${base}/docs`} className="docs-link">API docs ↗</a>
          <button className="theme-btn" onClick={onThemeToggle} title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}>
            {isDark ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </div>
    </header>
  );
}

// ── Tab bar ────────────────────────────────────────────────────────────────
function TabBar({ active, onChange }) {
  const tabs = [
    { id: 'existing', label: 'Existing User' },
    { id: 'newuser',  label: 'New User' },
    { id: 'similar',  label: 'Similar Recipes' },
    { id: 'metrics',  label: 'Metrics' },
  ];
  return (
    <div className="tab-bar">
      {tabs.map(t => (
        <button key={t.id} className={`tab-btn${active === t.id ? ' active' : ''}`} onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

Object.assign(window, { Badge, ScoreBar, SkeletonCard, AnimatedNum, Toggle, Stepper, Header, TabBar });
