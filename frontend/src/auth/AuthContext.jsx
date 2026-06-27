import { createContext, useContext, useState, useCallback, useEffect } from 'react'

/**
 * Authentication context.
 * Holds the bearer token + user in localStorage, exposes login/logout, and an
 * `authedFetch` wrapper that attaches the token and force-logs-out on a 401.
 */
const AuthContext = createContext(null)
const TOKEN_KEY = 'bct-token'
const USER_KEY = 'bct-user'

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || null)
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem(USER_KEY)) } catch { return null }
  })
  // Until an existing token is verified we show a brief loading state.
  const [loading, setLoading] = useState(!!localStorage.getItem(TOKEN_KEY))

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }, [])

  const login = useCallback(async (username, password) => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data.detail || 'Login failed')
    setToken(data.token)
    setUser(data.user)
    localStorage.setItem(TOKEN_KEY, data.token)
    localStorage.setItem(USER_KEY, JSON.stringify(data.user))
    return data.user
  }, [])

  const authedFetch = useCallback(async (url, options = {}) => {
    const headers = { ...(options.headers || {}) }
    if (token) headers.Authorization = `Bearer ${token}`
    if (options.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json'
    const res = await fetch(url, { ...options, headers })
    if (res.status === 401) logout()      // expired/invalid -> back to login
    return res
  }, [token, logout])

  // Verify a persisted token once on mount; refresh the cached user record.
  useEffect(() => {
    if (!token) { setLoading(false); return }
    let cancelled = false
    fetch('/api/auth/me', { headers: { Authorization: `Bearer ${token}` } })
      .then(r => { if (!r.ok) throw new Error('invalid'); return r.json() })
      .then(u => {
        if (cancelled) return
        setUser(u)
        localStorage.setItem(USER_KEY, JSON.stringify(u))
      })
      .catch(() => { if (!cancelled) logout() })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const value = {
    token, user, loading,
    isAuthenticated: !!token,
    isAdmin: user?.role === 'admin',
    login, logout, authedFetch,
  }
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}
