import { Building2, Search, GitBranch, MessageSquare, Clock, Zap, Network, Sun, Moon, Monitor } from 'lucide-react'

const STATS = [
  { value: '450+', label: 'Documents' },
  { value: '2.3k', label: 'Relations' },
  { value: '3', label: 'Languages' },
  { value: 'Verified', label: 'Sources' },
]

export default function Sidebar({ mode, onModeChange, theme, onThemeChange, history, onHistoryClick }) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <Building2 size={20} strokeWidth={2.5} />
        </div>
        <div className="sidebar-brand-text">
          <h1>BCT Intel-Graph</h1>
          <span>Hybrid Graph RAG · PFE 2026</span>
        </div>
      </div>

      {/* Theme selection */}
      <div className="sidebar-label">Display Theme</div>
      <div className="theme-picker">
        <button 
          className={`theme-picker-btn ${theme === 'light' ? 'active' : ''}`}
          onClick={() => onThemeChange('light')}
          title="Light Theme"
        >
          <Sun size={14} />
        </button>
        <button 
          className={`theme-picker-btn ${theme === 'dark' ? 'active' : ''}`}
          onClick={() => onThemeChange('dark')}
          title="Dark Theme"
        >
          <Moon size={14} />
        </button>
        <button 
          className={`theme-picker-btn ${theme === 'system' ? 'active' : ''}`}
          onClick={() => onThemeChange('system')}
          title="System Preference"
        >
          <Monitor size={14} />
        </button>
      </div>

      {/* Mode section */}
      <div className="sidebar-label">Search Mode</div>
      <div className="mode-group">
        <button
          id="mode-rag-btn"
          className={`mode-btn ${mode === 'rag' ? 'active' : ''}`}
          onClick={() => onModeChange('rag')}
        >
          <Search size={15} />
          <div>
            <div>Standard RAG</div>
          </div>
        </button>
        <button
          id="mode-graph-btn"
          className={`mode-btn graph ${mode === 'graph' ? 'active' : ''}`}
          onClick={() => onModeChange('graph')}
        >
          <Network size={15} />
          <div>
            <div>Graph RAG</div>
          </div>
        </button>
      </div>

      {/* History */}
      <div className="sidebar-label" style={{ marginTop: 8 }}>History</div>
      <div className="history-list">
        {history.length === 0 && (
          <div style={{ padding: '8px 12px', fontSize: 12, color: 'var(--text-muted)' }}>
            Your questions will appear here.
          </div>
        )}
        {[...history].reverse().slice(0, 15).map(msg => (
          <div
            key={msg.id}
            className="history-item"
            onClick={() => onHistoryClick(msg.text)}
            title={msg.text}
          >
            <Clock size={12} style={{ flexShrink: 0 }} />
            <span className="history-item-text">{msg.text}</span>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="sidebar-label">Corpus Statistics</div>
      <div className="stats-grid">
        {STATS.map(s => (
          <div key={s.label} className="stat-card">
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>
    </aside>
  )
}
