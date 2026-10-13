import { useEffect, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, BrainCircuit, Check, CircleHelp, Clock3, MessageSquareText, Send, Sparkles, TrendingUp } from 'lucide-react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const initialExamples = [
  { text: 'Just tried the new update and it is so much faster. Love it!', sentiment: 'positive', confidence: .97, timestamp: new Date().toISOString() },
  { text: 'Been waiting two weeks and still no response from support.', sentiment: 'negative', confidence: .94, timestamp: new Date(Date.now()-3600000).toISOString() },
  { text: 'The packaging is beautiful, but the product itself feels cheap.', sentiment: 'negative', confidence: .82, timestamp: new Date(Date.now()-7200000).toISOString() },
]
const shortTime = value => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function App() {
  const [data, setData] = useState({ total: 0, positive: 0, negative: 0, timeline: [], recent: [] })
  const [text, setText] = useState('')
  const [examples, setExamples] = useState(initialExamples)
  const [loading, setLoading] = useState(false)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')

  async function refresh() {
    try {
      const response = await fetch(`${API}/api/trends?hours=24`)
      if (!response.ok) throw new Error('API is starting up or the model is unavailable.')
      const result = await response.json()
      setData(result); setConnected(true); setError('')
    } catch { setConnected(false) }
  }
  useEffect(() => { refresh(); const timer = setInterval(refresh, 5000); return () => clearInterval(timer) }, [])

  async function analyze(event) {
    event.preventDefault()
    if (!text.trim()) return
    setLoading(true); setError('')
    try {
      const response = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Could not analyze this text.')
      setExamples(current => [result, ...current].slice(0, 8)); setText(''); await refresh()
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  const positivePct = data.total ? Math.round(data.positive / data.total * 100) : 0
  const displayTimeline = data.timeline.length ? data.timeline : Array.from({ length: 8 }, (_, i) => ({ hour: new Date(Date.now()-(7-i)*3600000).toISOString(), positive: [8,12,10,17,14,22,19,27][i], negative: [5,7,8,6,10,8,12,9][i] }))
  const messages = data.recent.length ? data.recent : examples
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon"><Activity size={19}/></div><span>signal<span className="brand-dot">.</span></span></div>
      <div className="workspace"><div className="workspace-avatar">S</div><div><b>Social listening</b><small>Workspace</small></div><span className="chevron">⌄</span></div>
      <div className="nav-label">WORKSPACE</div>
      <nav><a className="active"><BarChart3 size={17}/> Overview <span className="nav-pip"/></a><a><MessageSquareText size={17}/> Mentions <span className="nav-count">24</span></a><a><TrendingUp size={17}/> Topics</a></nav>
      <div className="nav-label sources-label">SOURCES</div><div className="source-row"><span className="source-x">𝕏</span><span>X / Twitter</span><span className="live-dot"/></div><div className="source-row"><span className="source-reddit">●</span><span>Reddit</span><span className="muted-dot"/></div>
      <div className="sidebar-bottom"><div className="help"><CircleHelp size={16}/> Help center <ArrowUpRight size={13}/></div><div className="profile"><div className="profile-avatar">KH</div><div><b>Kazi Hossain</b><small>Free plan</small></div><span className="dots">···</span></div></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <b>Overview</b></div><div className="top-actions"><span className="period"><Clock3 size={14}/> Last 24 hours <span>⌄</span></span><div className="connection"><i className={connected ? 'connected' : ''}/>{connected ? 'Live data' : 'Connecting'}</div><div className="top-avatar">KH</div></div></header>
      <section className="page-heading"><div><div className="eyebrow"><span/> SOCIAL SENTIMENT</div><h1>Good morning, Kazi <span className="wave">✳</span></h1><p>Here's what's happening across your social channels today.</p></div><button className="export-btn" onClick={() => window.print()}><ArrowUpRight size={15}/> Export report</button></section>
      <section className="stat-grid">
        <article className="stat-card"><div className="stat-top"><span>Total mentions</span><span className="stat-icon purple"><MessageSquareText size={16}/></span></div><div className="stat-number">{data.total ? data.total.toLocaleString() : '2,847'} <span className="change positive-change"><ArrowUpRight size={13}/> 12.8%</span></div><div className="stat-foot">vs. previous 24 hours</div><div className="mini-bars">{[30,44,36,58,45,70,60,83,68,91,76,100].map((h,i)=><i key={i} style={{height:`${h}%`}}/>)}</div></article>
        <article className="stat-card"><div className="stat-top"><span>Positive sentiment</span><span className="stat-icon green"><Sparkles size={16}/></span></div><div className="stat-number">{data.total ? positivePct : 68}<small>%</small> <span className="change positive-change"><ArrowUpRight size={13}/> 4.2%</span></div><div className="sentiment-bar"><i style={{width:`${data.total ? positivePct : 68}%`}}/></div><div className="stat-foot">{data.positive.toLocaleString() || '1,936'} positive mentions</div></article>
        <article className="stat-card"><div className="stat-top"><span>Negative sentiment</span><span className="stat-icon red"><Activity size={16}/></span></div><div className="stat-number">{data.total ? 100-positivePct : 32}<small>%</small> <span className="change negative-change">↓ 2.1%</span></div><div className="sentiment-bar negative-track"><i style={{width:`${data.total ? 100-positivePct : 32}%`}}/></div><div className="stat-foot">{data.negative.toLocaleString() || '911'} negative mentions</div></article>
        <article className="stat-card score-card"><div className="stat-top"><span>Sentiment score</span><span className="stat-icon blue"><TrendingUp size={16}/></span></div><div className="stat-number">{data.total ? Math.round(positivePct-data.negative/data.total*100) : '+36'} <span className="score-label">/ 100</span></div><div className="score-scale"><i/><span>−100</span><span>0</span><span>+100</span></div><div className="stat-foot">Overall brand perception</div></article>
      </section>
      <section className="middle-grid">
        <article className="panel chart-panel"><div className="panel-heading"><div><h2>Sentiment over time</h2><p>Volume of positive and negative mentions</p></div><button className="select-btn">Hourly <span>⌄</span></button></div><div className="chart-legend"><span><i className="legend-positive"/> Positive</span><span><i className="legend-negative"/> Negative</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={displayTimeline} margin={{top:8,right:8,left:-22,bottom:0}}><defs><linearGradient id="posFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#7770ee" stopOpacity={.2}/><stop offset="100%" stopColor="#7770ee" stopOpacity={0}/></linearGradient><linearGradient id="negFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#f28c78" stopOpacity={.15}/><stop offset="100%" stopColor="#f28c78" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#eef0f4" vertical={false}/><XAxis dataKey="hour" tickFormatter={v=>shortTime(v)} axisLine={false} tickLine={false} tick={{fill:'#9ba1ad',fontSize:11}} minTickGap={28}/><YAxis axisLine={false} tickLine={false} tick={{fill:'#9ba1ad',fontSize:11}}/><Tooltip labelFormatter={v=>new Date(v).toLocaleString()} contentStyle={{border:'1px solid #ececf2',borderRadius:10,fontSize:12}}/><Area type="monotone" dataKey="positive" stroke="#7770ee" strokeWidth={2.5} fill="url(#posFill)"/><Area type="monotone" dataKey="negative" stroke="#f28c78" strokeWidth={2.5} fill="url(#negFill)"/></AreaChart></ResponsiveContainer></div></article>
        <article className="panel breakdown-panel"><div className="panel-heading"><div><h2>Sentiment breakdown</h2><p>Share of total mentions</p></div><button className="more-btn">···</button></div><div className="donut-area"><div className="donut"><div className="donut-center"><b>{data.total ? data.total.toLocaleString() : '2,847'}</b><span>mentions</span></div></div></div><div className="breakdown-rows"><div><span><i className="legend-positive"/> Positive</span><b>{data.total ? positivePct : 68}%</b></div><div><span><i className="legend-neutral"/> Neutral</span><b>0%</b></div><div><span><i className="legend-negative"/> Negative</span><b>{data.total ? 100-positivePct : 32}%</b></div></div></article>
      </section>
      <section className="bottom-grid">
        <article className="panel mentions-panel"><div className="panel-heading"><div><h2>Recent mentions</h2><p>Latest conversations about your brand</p></div><button className="view-all">View all <ArrowUpRight size={13}/></button></div><div className="mentions-list">{messages.slice(0,4).map((item,i)=><div className="mention" key={`${item.timestamp}-${i}`}><div className={`mention-avatar avatar-${i%4}`}>{['𝕏','r','in','✳'][i%4]}</div><div className="mention-content"><div className="mention-meta"><b>{['@alexmorgan','u/techobserver','@jamie_w','@producthunt'][i%4]}</b><span>·</span><time>{shortTime(item.timestamp)}</time><span className={`sentiment-tag ${item.sentiment}`}>{item.sentiment}</span></div><p>{item.text}</p></div></div>)}</div></article>
        <article className="panel analyze-panel"><div className="analyze-title"><div className="analyze-icon"><BrainCircuit size={17}/></div><div><h2>Analyze sentiment</h2><p>Try it with any message</p></div><span className="powered">BERT</span></div><form onSubmit={analyze}><textarea value={text} onChange={e=>setText(e.target.value)} placeholder="Paste a social post or message to analyze..." maxLength={10000}/><div className="form-footer"><span>{text.length}/10,000</span><button disabled={loading || !text.trim()}><Send size={14}/>{loading?'Analyzing':'Analyze'}</button></div></form>{error&&<div className="error-message">{error}</div>}<div className="model-note"><Check size={13}/> Powered by fine-tuned BERT <span>·</span> Confidence scores included</div></article>
      </section>
      <footer>Sentiment data refreshes automatically <span>·</span> Last synced just now <i className={connected ? 'connected' : ''}/></footer>
    </main>
  </div>
}

export default App
