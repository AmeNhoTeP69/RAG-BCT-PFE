import { useState } from 'react'
import { Building2, Lock, User, LogIn, Loader2 } from 'lucide-react'
import { useAuth } from '../auth/AuthContext'

export default function LoginPage() {
  const { login } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (!username.trim() || !password || busy) return
    setBusy(true)
    setError('')
    try {
      await login(username.trim(), password)
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-screen">
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <form className="login-card glass-card" onSubmit={submit}>
        <div className="login-brand">
          <div className="login-brand-icon">
            <Building2 size={28} strokeWidth={2.5} />
          </div>
          <h1>BCT Intel-Graph</h1>
          <span>Hybrid Graph RAG · Central Bank of Tunisia</span>
        </div>

        <label className="login-field">
          <User size={16} />
          <input
            type="text"
            value={username}
            onChange={e => setUsername(e.target.value)}
            placeholder="Username"
            autoComplete="username"
            autoFocus
          />
        </label>

        <label className="login-field">
          <Lock size={16} />
          <input
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            placeholder="Password"
            autoComplete="current-password"
          />
        </label>

        {error && <div className="login-error">{error}</div>}

        <button type="submit" className="login-btn" disabled={busy || !username.trim() || !password}>
          {busy ? <Loader2 size={16} className="spin" /> : <LogIn size={16} />}
          {busy ? 'Signing in…' : 'Sign in'}
        </button>

        <div className="login-hint">Restricted access — secure BCT area</div>
      </form>
    </div>
  )
}
