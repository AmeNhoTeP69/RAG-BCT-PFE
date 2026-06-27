import { useState, useEffect } from 'react'
import { Building2, Loader2 } from 'lucide-react'
import { useAuth } from './auth/AuthContext'
import LoginPage from './components/LoginPage'
import MainApp from './MainApp'

export default function App() {
  const { isAuthenticated, loading } = useAuth()

  // Theme state: 'light' | 'dark' | 'system' — applied for login AND main app.
  const [theme, setTheme] = useState(() => localStorage.getItem('bct-theme') || 'system')

  useEffect(() => {
    const root = window.document.body
    const darkQuery = window.matchMedia('(prefers-color-scheme: dark)')

    const applyTheme = () => {
      localStorage.setItem('bct-theme', theme)
      root.classList.remove('light-theme')
      const isDark = theme === 'dark' || (theme === 'system' && darkQuery.matches)
      if (!isDark) root.classList.add('light-theme')
    }

    applyTheme()
    const listener = () => { if (theme === 'system') applyTheme() }
    darkQuery.addEventListener('change', listener)
    return () => darkQuery.removeEventListener('change', listener)
  }, [theme])

  if (loading) {
    return (
      <div className="login-screen">
        <div className="bg-orb bg-orb-1" />
        <div className="auth-loading">
          <div className="login-brand-icon"><Building2 size={28} strokeWidth={2.5} /></div>
          <Loader2 size={20} className="spin" />
          <span>Chargement…</span>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) return <LoginPage />

  return <MainApp theme={theme} onThemeChange={setTheme} />
}
