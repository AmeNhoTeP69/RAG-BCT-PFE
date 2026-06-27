import { useRef, useEffect, useState } from 'react'
import { Send, Building2, Zap } from 'lucide-react'
import Message from './Message'
import ThinkingBubble from './ThinkingBubble'

const EXAMPLE_QUERIES = [
  'What are the internal control rules for managing AML risk?',
  'ما هي شروط فتح مكتب صرف في تونس؟',
  'How is Law 2015-26 related to bank internal control?',
  'What are the reporting requirements for foreign exchange operations?',
]

export default function ChatWindow({ messages, isLoading, onSend, onMsgSelect, selectedMsgId }) {
  const [input, setInput] = useState('')
  const textareaRef = useRef(null)
  const bottomRef = useRef(null)

  // Auto-scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [input])

  const handleSend = () => {
    if (!input.trim() || isLoading) return
    onSend(input.trim())
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <main className="chat-window">
      {/* Top Nav */}
      <header className="chat-topnav">
        <div className="chat-topnav-title">
          <h2>Central Bank of Tunisia</h2>
          <span>Intelligent Regulatory Explorer — Circulars &amp; Notes</span>
        </div>
        <div className="status-pill" id="system-status">
          <span className="status-dot" />
          System Connected
        </div>
      </header>

      {/* Messages */}
      <div className="messages-list">
        {messages.length === 0 && (
          <div className="welcome-card glass-card">
            <div className="welcome-icon">
              <Building2 size={28} strokeWidth={2} color="#080b14" />
            </div>
            <h2>BCT Intel-Graph</h2>
            <p>
              Ask questions about <strong>circulars</strong> and{' '}
              <strong>regulatory notes</strong> from the Central Bank of Tunisia.
              Get precise answers based exclusively on the official document corpus.
            </p>
            <div className="welcome-chips">
              {EXAMPLE_QUERIES.map((q, i) => (
                <button
                  key={i}
                  className="welcome-chip"
                  onClick={() => onSend(q)}
                  id={`example-query-${i}`}
                >
                  {q.length > 60 ? q.slice(0, 57) + '…' : q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <Message
            key={msg.id}
            msg={msg}
            isSelected={msg.id === selectedMsgId}
            onSelect={() => onMsgSelect(msg)}
          />
        ))}

        {isLoading && <ThinkingBubble />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            id="chat-input"
            className="chat-textarea"
            rows={1}
            placeholder="Ask your regulatory question… (English, French or Arabic)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            id="send-btn"
            className="send-btn"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            title="Send (Enter)"
          >
            <Send size={16} strokeWidth={2.5} />
          </button>
        </div>
        <div className="input-hint">
          <Zap size={11} style={{ display: 'inline', marginRight: 4, color: 'var(--accent)' }} />
          Press Enter to send your query
        </div>
      </div>
    </main>
  )
}
