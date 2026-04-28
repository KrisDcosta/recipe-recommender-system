const { useState, useEffect } = React;

function App() {
  const [tab,             setTab]      = useState('existing');
  const [health,          setHealth]   = useState('unknown');
  const [similarRecipeId, setSimilarId]= useState(null);
  const [isDark,          setIsDark]   = useState(() => {
    const stored = localStorage.getItem('rr-theme');
    return stored ? stored === 'dark' : true;
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('rr-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  useEffect(() => { api.health().then(setHealth); }, []);

  function handleFindSimilar(recipeId) {
    setSimilarId(recipeId);
    setTab('similar');
  }

  return (
    <>
      <Header health={health} isDark={isDark} onThemeToggle={() => setIsDark(d => !d)} />
      <TabBar active={tab} onChange={setTab} />
      <div className="content-area">
        {tab === 'existing' && <ExistingUserTab onFindSimilar={handleFindSimilar} />}
        {tab === 'newuser'  && <NewUserTab      onFindSimilar={handleFindSimilar} />}
        {tab === 'similar'  && <SimilarTab      initialRecipeId={similarRecipeId} key={similarRecipeId ?? 'empty'} />}
        {tab === 'metrics'  && <MetricsTab />}
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
