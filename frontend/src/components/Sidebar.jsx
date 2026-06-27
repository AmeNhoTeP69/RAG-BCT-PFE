import {
  Building2, Search, Network, Sun, Moon, Monitor, Plus, Trash2,
  MessageSquare, Users, BarChart3, LogOut, BookCheck,
} from 'lucide-react'

function relativeTime(iso) {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const s = Math.max(0, (Date.now() - then) / 1000)
  if (s < 60) return 'just now'
  const m = Math.floor(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.floor(h / 24)
  if (d < 30) return `${d} d ago`
  return new Date(iso).toLocaleDateString('en-GB')
}

export default function Sidebar({
  mode, onModeChange, theme, onThemeChange,
  user, isAdmin, view, onNavigate,
  conversations, currentConvId, onNewConversation, onSelectConversation, onDeleteConversation,
  onLogout,
}) {
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

      {/* New conversation */}
      <button className="new-conv-btn" onClick={onNewConversation}>
        <Plus size={16} strokeWidth={2.5} />
        New conversation
      </button>

      {/* Primary navigation (admin gets extra destinations) */}
      <div className="nav-group">
        <button
          className={`nav-item ${view === 'chat' ? 'active' : ''}`}
          onClick={() => onNavigate('chat')}
        >
          <MessageSquare size={15} /> Assistant
        </button>
        {isAdmin && (
          <>
            <button
              className={`nav-item ${view === 'users' ? 'active' : ''}`}
              onClick={() => onNavigate('users')}
            >
              <Users size={15} /> Account management
            </button>
            <button
              className={`nav-item ${view === 'dashboard' ? 'active' : ''}`}
              onClick={() => onNavigate('dashboard')}
            >
              <BarChart3 size={15} /> Dashboard
            </button>
            <button
              className={`nav-item ${view === 'corrections' ? 'active' : ''}`}
              onClick={() => onNavigate('corrections')}
            >
              <BookCheck size={15} /> Corrections
            </button>
          </>
        )}
      </div>

      {/* Search mode */}
      <div className="sidebar-label">Search mode</div>
      <div className="mode-group">
        <button
          className={`mode-btn ${mode === 'rag' ? 'active' : ''}`}
          onClick={() => onModeChange('rag')}
        >
          <Search size={15} /> <div><div>Standard RAG</div></div>
        </button>
        <button
          className={`mode-btn graph ${mode === 'graph' ? 'active' : ''}`}
          onClick={() => onModeChange('graph')}
        >
          <Network size={15} /> <div><div>Graph RAG</div></div>
        </button>
      </div>

      {/* Conversations */}
      <div className="sidebar-label" style={{ marginTop: 6 }}>Conversations</div>
      <div className="history-list">
        {(!conversations || conversations.length === 0) && (
          <div style={{ padding: '8px 12px', fontSize: 12, color: 'var(--text-muted)' }}>
            Your conversations will appear here.
          </div>
        )}
        {conversations && conversations.map(conv => (
          <div
            key={conv.id}
            className={`conv-item ${conv.id === currentConvId ? 'active' : ''}`}
            onClick={() => onSelectConversation(conv.id)}
            title={conv.title}
          >
            <MessageSquare size={13} className="conv-item-icon" />
            <div className="conv-item-body">
              <span className="conv-item-title">{conv.title}</span>
              <span className="conv-item-time">{relativeTime(conv.last_message_at || conv.updated_at)}</span>
            </div>
            <button
              className="conv-item-del"
              title="Delete"
              onClick={(e) => { e.stopPropagation(); onDeleteConversation(conv.id) }}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>

      {/* Theme */}
      <div className="theme-picker" style={{ marginTop: 6 }}>
        <button className={`theme-picker-btn ${theme === 'light' ? 'active' : ''}`}
          onClick={() => onThemeChange('light')} title="Light theme"><Sun size={14} /></button>
        <button className={`theme-picker-btn ${theme === 'dark' ? 'active' : ''}`}
          onClick={() => onThemeChange('dark')} title="Dark theme"><Moon size={14} /></button>
        <button className={`theme-picker-btn ${theme === 'system' ? 'active' : ''}`}
          onClick={() => onThemeChange('system')} title="System preference"><Monitor size={14} /></button>
      </div>

      {/* User footer + logout */}
      <div className="sidebar-user">
        <div className="sidebar-user-info">
          <div className="sidebar-user-avatar">{(user?.username || '?').slice(0, 1).toUpperCase()}</div>
          <div className="sidebar-user-text">
            <span className="sidebar-user-name">{user?.username}</span>
            <span className="sidebar-user-sub">
              {user?.role === 'admin' ? 'Administrator' : (user?.bank_name || 'Bank')}
            </span>
          </div>
        </div>
        <button className="logout-btn" onClick={onLogout} title="Log out">
          <LogOut size={15} />
        </button>
      </div>
    </aside>
  )
}
