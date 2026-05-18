import { useState, useCallback, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import SourcePanel from './components/SourcePanel'

export default function App() {
  const [mode, setMode] = useState('rag')          // 'rag' | 'graph'
  const [messages, setMessages] = useState([])     // { id, role, text, sources, relatedNodes, chunks }
  const [isLoading, setIsLoading] = useState(false)
  const [selectedMsg, setSelectedMsg] = useState(null) // message object for right panel
  
  // Theme state: 'light' | 'dark' | 'system'
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('bct-theme') || 'system';
  });

  useEffect(() => {
    const root = window.document.body;
    const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const applyTheme = () => {
      localStorage.setItem('bct-theme', theme);
      root.classList.remove('light-theme');
      
      const isDark = theme === 'dark' || (theme === 'system' && darkQuery.matches);
      if (!isDark) {
        root.classList.add('light-theme');
      }
    };

    applyTheme();
    
    const listener = () => {
      if (theme === 'system') applyTheme();
    };
    
    darkQuery.addEventListener('change', listener);
    return () => darkQuery.removeEventListener('change', listener);
  }, [theme]);

  const sendQuery = useCallback(async (query) => {
    if (!query.trim() || isLoading) return

    const userMsg = { id: Date.now(), role: 'user', text: query }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setSelectedMsg(null)

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      const aiMsg = {
        id: Date.now() + 1,
        role: 'ai',
        text: data.answer ?? 'Aucune réponse reçue.',
        sources: data.sources ?? [],
        relatedNodes: data.related_nodes ?? [],
        chunks: data.retrieved_chunks ?? [],
        mode,
      }
      setMessages(prev => [...prev, aiMsg])
      setSelectedMsg(aiMsg)
    } catch (err) {
      const errMsg = {
        id: Date.now() + 1,
        role: 'ai',
        text: "⚠️ Erreur de connexion au serveur. Vérifiez que `python api.py` est en cours d'exécution.",
        sources: [], relatedNodes: [], chunks: [], mode,
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setIsLoading(false)
    }
  }, [mode, isLoading])

  return (
    <div className="app-layout">
      {/* Decorative background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <Sidebar
        mode={mode}
        onModeChange={setMode}
        theme={theme}
        onThemeChange={setTheme}
        history={messages.filter(m => m.role === 'user')}
        onHistoryClick={(text) => sendQuery(text)}
      />

      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onSend={sendQuery}
        onMsgSelect={setSelectedMsg}
        selectedMsgId={selectedMsg?.id}
      />

      <SourcePanel message={selectedMsg} />
    </div>
  )
}
