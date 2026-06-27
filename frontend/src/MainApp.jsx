import { useState, useCallback, useEffect } from 'react'
import { useAuth } from './auth/AuthContext'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import SourcePanel from './components/SourcePanel'
import UserManagement from './components/UserManagement'
import Dashboard from './components/Dashboard'
import CorrectionsPanel from './components/CorrectionsPanel'

/**
 * Authenticated experience: sidebar + (chat | admin views).
 * Conversations are persisted server-side via /api/chat and listed in the sidebar.
 */
export default function MainApp({ theme, onThemeChange }) {
  const { user, isAdmin, logout, authedFetch } = useAuth()

  const [mode, setMode] = useState('rag')              // 'rag' | 'graph'
  const [view, setView] = useState('chat')             // 'chat' | 'users' | 'dashboard'
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedMsg, setSelectedMsg] = useState(null)
  const [conversations, setConversations] = useState([])
  const [currentConvId, setCurrentConvId] = useState(null)

  const loadConversations = useCallback(async () => {
    try {
      const res = await authedFetch('/api/conversations')
      if (res.ok) setConversations(await res.json())
    } catch { /* offline — keep current list */ }
  }, [authedFetch])

  useEffect(() => { loadConversations() }, [loadConversations])

  const newConversation = useCallback(() => {
    setCurrentConvId(null)
    setMessages([])
    setSelectedMsg(null)
    setView('chat')
  }, [])

  const loadConversation = useCallback(async (id) => {
    setView('chat')
    try {
      const res = await authedFetch(`/api/conversations/${id}`)
      if (!res.ok) return
      const data = await res.json()
      const msgs = (data.messages || []).map(m => ({
        id: m.id, role: m.role, text: m.text,
        sources: m.sources || [], relatedNodes: m.relatedNodes || [],
        chunks: m.chunks || [], mode: m.mode, cached: m.cached,
      }))
      setMessages(msgs)
      setCurrentConvId(id)
      setSelectedMsg(null)
    } catch { /* ignore */ }
  }, [authedFetch])

  const deleteConversation = useCallback(async (id) => {
    try { await authedFetch(`/api/conversations/${id}`, { method: 'DELETE' }) } catch { /* ignore */ }
    if (id === currentConvId) newConversation()
    loadConversations()
  }, [authedFetch, currentConvId, newConversation, loadConversations])

  const sendQuery = useCallback(async (query) => {
    if (!query.trim() || isLoading) return
    setView('chat')
    const userMsg = { id: `u-${Date.now()}`, role: 'user', text: query }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setSelectedMsg(null)
    try {
      const res = await authedFetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ query, mode, conversation_id: currentConvId }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const aiMsg = {
        id: data.message_id ?? `a-${Date.now()}`,
        role: 'ai',
        text: data.answer ?? 'Aucune réponse reçue.',
        sources: data.sources ?? [],
        relatedNodes: data.related_nodes ?? [],
        chunks: data.retrieved_chunks ?? [],
        mode: data.mode ?? mode,
        cached: data.cached ?? false,
      }
      setMessages(prev => [...prev, aiMsg])
      setSelectedMsg(aiMsg)
      if (data.conversation_id) setCurrentConvId(data.conversation_id)
      loadConversations()
    } catch {
      setMessages(prev => [...prev, {
        id: `e-${Date.now()}`, role: 'ai',
        text: "⚠️ Server connection error. Please try again.",
        sources: [], relatedNodes: [], chunks: [], mode,
      }])
    } finally {
      setIsLoading(false)
    }
  }, [mode, isLoading, currentConvId, authedFetch, loadConversations])

  return (
    <div className="app-layout">
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <Sidebar
        mode={mode} onModeChange={setMode}
        theme={theme} onThemeChange={onThemeChange}
        user={user} isAdmin={isAdmin}
        view={view} onNavigate={setView}
        conversations={conversations} currentConvId={currentConvId}
        onNewConversation={newConversation}
        onSelectConversation={loadConversation}
        onDeleteConversation={deleteConversation}
        onLogout={logout}
      />

      {view === 'chat' && (
        <>
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            onSend={sendQuery}
            onMsgSelect={setSelectedMsg}
            selectedMsgId={selectedMsg?.id}
          />
          <SourcePanel message={selectedMsg} />
        </>
      )}
      {view === 'users' && <UserManagement />}
      {view === 'dashboard' && <Dashboard />}
      {view === 'corrections' && <CorrectionsPanel />}
    </div>
  )
}
