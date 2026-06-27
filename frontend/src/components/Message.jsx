import ReactMarkdown from 'react-markdown'
import { Building2, User, FileText, Network, ChevronRight, Zap } from 'lucide-react'

// Detect if text is predominantly Arabic/RTL
function isRTL(text) {
  const rtlChars = (text || '').match(/[\u0600-\u06FF\u0750-\u077F]/g) || []
  return rtlChars.length > text.length * 0.25
}

export default function Message({ msg, isSelected, onSelect }) {
  const isUser = msg.role === 'user'
  const rtl = !isUser && isRTL(msg.text)

  return (
    <div className={`message-row ${isUser ? 'user' : 'ai'}`}>
      {/* Avatar */}
      <div className={`msg-avatar ${isUser ? 'user' : 'ai'}`}>
        {isUser
          ? <User size={16} strokeWidth={2.5} color="#fff" />
          : <Building2 size={16} strokeWidth={2.5} color="#080b14" />}
      </div>

      {/* Bubble */}
      <div style={{ display: 'flex', flexDirection: 'column', maxWidth: '68%' }}>
        <div
          className={`msg-bubble ${isUser ? 'user' : 'ai'}`}
          dir={rtl ? 'rtl' : 'ltr'}
          style={{ cursor: !isUser ? 'pointer' : 'default',
                   outline: isSelected ? '1px solid var(--accent)' : 'none' }}
          onClick={!isUser ? onSelect : undefined}
        >
          {isUser
            ? <span>{msg.text}</span>
            : <ReactMarkdown>{msg.text}</ReactMarkdown>}
        </div>

        {/* Cache indicator — answer served instantly from the semantic cache */}
        {!isUser && msg.cached && (
          <div className="cache-badge" title="Answer served instantly from the semantic cache">
            <Zap size={11} />
            Instant answer · cached
          </div>
        )}

        {/* Source tags */}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="msg-meta">
            {msg.sources.slice(0, 5).map((src, i) => (
              <span key={i} className="source-tag">
                <FileText size={10} />
                {src}
              </span>
            ))}
            {msg.relatedNodes && msg.relatedNodes.length > 0 && (
              <span className="source-tag graph">
                <Network size={10} />
                {msg.relatedNodes.length} associated links
              </span>
            )}
            <button className="view-evidence-btn" onClick={onSelect}>
              <ChevronRight size={11} />
              View sources
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
