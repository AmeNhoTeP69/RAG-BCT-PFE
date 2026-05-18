import { FileText, Network, BookOpen, GitBranch, MousePointerClick } from 'lucide-react'

// Detect RTL
function isRTL(text) {
  const rtlChars = (text || '').match(/[\u0600-\u06FF]/g) || []
  return rtlChars.length > (text?.length || 1) * 0.2
}

export default function SourcePanel({ message }) {
  const hasSources  = message?.sources?.length > 0
  const hasNodes    = message?.relatedNodes?.length > 0
  const hasChunks   = message?.chunks?.length > 0
  const hasAnything = hasSources || hasNodes || hasChunks

  if (!message || message.role !== 'ai') {
    return (
      <aside className="source-panel">
        <div className="panel-header">
        <h3>Evidence Panel</h3>
        <span>Sources · Graph · Excerpts</span>
      </div>
      <div className="panel-empty">
        <div className="panel-empty-icon">
          <MousePointerClick size={22} color="var(--text-muted)" />
        </div>
        <p>Click on a response to view the sources and document evidence.</p>
        <span>Document excerpts and graph nodes will appear here.</span>
      </div>
      </aside>
    )
  }

  return (
    <aside className="source-panel">
      <div className="panel-header">
        <h3>Evidence Panel</h3>
        <span>
          {message.mode === 'graph' ? '🕸️ Graph RAG Mode' : '🔍 Standard RAG Mode'}
          {hasSources ? ` · ${message.sources.length} source(s)` : ''}
        </span>
      </div>

      {/* Sources */}
      {hasSources && (
        <div className="panel-section">
          <div className="panel-section-title">
            <FileText size={11} />
            Source Documents
          </div>
          {message.sources.map((src, i) => (
            <div key={i} className="source-card">
              <div className="source-card-icon">
                <FileText size={14} />
              </div>
              <div className="source-card-text">
                <div className="source-card-name">{src}</div>
                <div className="source-card-sub">Circular · BCT</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Graph nodes */}
      {hasNodes && (
        <div className="panel-section">
          <div className="panel-section-title">
            <Network size={11} />
            Graph Entities &amp; Relations
          </div>
          <div>
            {message.relatedNodes.map((node, i) => (
              <span key={i} className="graph-node-chip">
                <GitBranch size={10} />
                {node}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Evidence chunks */}
      {hasChunks && (
        <div className="panel-section">
          <div className="panel-section-title">
            <BookOpen size={11} />
            Document Excerpts ({message.chunks.length})
          </div>
          {message.chunks.map((chunk, i) => {
            const title   = chunk.metadata?.title || chunk.metadata?.source || `Extrait ${i + 1}`
            const header  = chunk.metadata?.section_header || ''
            const text    = chunk.text || ''
            const rtl     = isRTL(text)
            return (
              <div key={i} className="evidence-item">
                <div className="evidence-item-header">
                  <FileText size={11} />
                  {title}{header ? ` — ${header}` : ''}
                </div>
                <div
                  className="evidence-item-text"
                  dir={rtl ? 'rtl' : 'ltr'}
                >
                  {text.length > 400 ? text.slice(0, 400) + '…' : text}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {!hasAnything && (
        <div className="panel-empty">
          <div className="panel-empty-icon">
            <FileText size={22} color="var(--text-muted)" />
          </div>
          <p>No sources available for this response.</p>
        </div>
      )}
    </aside>
  )
}
