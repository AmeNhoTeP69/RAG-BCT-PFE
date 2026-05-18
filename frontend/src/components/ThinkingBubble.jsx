import { Building2 } from 'lucide-react'

export default function ThinkingBubble() {
  return (
    <div className="thinking-bubble">
      <div className="msg-avatar ai">
        <Building2 size={16} strokeWidth={2.5} color="#080b14" />
      </div>
      <div className="thinking-dots">
        <span className="thinking-dot" />
        <span className="thinking-dot" />
        <span className="thinking-dot" />
        <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 6 }}>
          Searching...
        </span>
      </div>
    </div>
  )
}
