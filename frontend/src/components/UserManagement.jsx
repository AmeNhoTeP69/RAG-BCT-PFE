import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../auth/AuthContext'
import {
  UserPlus, Pencil, Trash2, Ban, CheckCircle2, X, Loader2, ShieldCheck,
} from 'lucide-react'

const EMPTY = { username: '', password: '', bank_name: '', role: 'bank' }

export default function UserManagement() {
  const { authedFetch, user: me } = useAuth()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [form, setForm] = useState(null)   // null = closed
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await authedFetch('/api/admin/users')
      if (res.ok) setUsers(await res.json())
    } finally { setLoading(false) }
  }, [authedFetch])

  useEffect(() => { load() }, [load])

  const isDefaultAdmin = (u) => u.username === 'admin'
  const isSelf = (u) => u.id === me?.id
  const locked = (u) => isDefaultAdmin(u) || isSelf(u)

  const openCreate = () => { setError(''); setForm({ mode: 'create', ...EMPTY }) }
  const openEdit = (u) => {
    setError('')
    setForm({ mode: 'edit', id: u.id, username: u.username, password: '', bank_name: u.bank_name || '', role: u.role })
  }

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      let res
      if (form.mode === 'create') {
        res = await authedFetch('/api/admin/users', {
          method: 'POST',
          body: JSON.stringify({
            username: form.username, password: form.password,
            bank_name: form.bank_name, role: form.role,
          }),
        })
      } else {
        const body = { bank_name: form.bank_name, role: form.role }
        if (form.password) body.password = form.password
        res = await authedFetch(`/api/admin/users/${form.id}`, { method: 'PUT', body: JSON.stringify(body) })
      }
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || 'Something went wrong')
      setForm(null)
      load()
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  const toggleStatus = async (u) => {
    const status = u.status === 'active' ? 'suspended' : 'active'
    const res = await authedFetch(`/api/admin/users/${u.id}/status`, { method: 'POST', body: JSON.stringify({ status }) })
    if (!res.ok) { const d = await res.json().catch(() => ({})); window.alert(d.detail || 'Error') }
    load()
  }

  const remove = async (u) => {
    if (!window.confirm(`Permanently delete the account “${u.username}”?`)) return
    const res = await authedFetch(`/api/admin/users/${u.id}`, { method: 'DELETE' })
    if (!res.ok) { const d = await res.json().catch(() => ({})); window.alert(d.detail || 'Error') }
    load()
  }

  return (
    <main className="admin-page">
      <div className="admin-header">
        <div>
          <h1 className="admin-title">Account management</h1>
          <p className="admin-subtitle">Create and manage bank and administrator accounts.</p>
        </div>
        <button className="admin-primary-btn" onClick={openCreate}>
          <UserPlus size={16} /> New account
        </button>
      </div>

      {loading ? (
        <div className="admin-empty"><Loader2 size={22} className="spin" /></div>
      ) : (
        <div className="admin-table-wrap glass-card">
          <table className="admin-table">
            <thead>
              <tr>
                <th>User</th><th>Bank</th><th>Role</th>
                <th>Status</th><th>Created</th><th></th>
              </tr>
            </thead>
            <tbody>
              {users.map(u => (
                <tr key={u.id}>
                  <td>
                    <div className="user-cell">
                      <div className="user-cell-avatar">{u.username.slice(0, 1).toUpperCase()}</div>
                      <span>{u.username}</span>
                    </div>
                  </td>
                  <td>{u.bank_name || '—'}</td>
                  <td>
                    <span className={`badge role-${u.role}`}>
                      {u.role === 'admin' ? <><ShieldCheck size={11} /> Admin</> : 'Bank'}
                    </span>
                  </td>
                  <td><span className={`badge status-${u.status}`}>{u.status === 'active' ? 'Active' : 'Suspended'}</span></td>
                  <td className="muted-cell">{new Date(u.created_at).toLocaleDateString('en-GB')}</td>
                  <td>
                    <div className="row-actions">
                      <button onClick={() => openEdit(u)} title="Edit"><Pencil size={15} /></button>
                      <button onClick={() => toggleStatus(u)} disabled={locked(u)}
                        title={u.status === 'active' ? 'Suspend' : 'Reactivate'}>
                        {u.status === 'active' ? <Ban size={15} /> : <CheckCircle2 size={15} />}
                      </button>
                      <button onClick={() => remove(u)} disabled={locked(u)} className="danger" title="Delete">
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {form && (
        <div className="modal-overlay" onClick={() => setForm(null)}>
          <form className="modal-card glass-card" onClick={e => e.stopPropagation()} onSubmit={submit}>
            <div className="modal-header">
              <h3>{form.mode === 'create' ? 'New account' : `Edit “${form.username}”`}</h3>
              <button type="button" onClick={() => setForm(null)}><X size={18} /></button>
            </div>

            {form.mode === 'create' && (
              <label className="modal-field">
                <span>Username</span>
                <input value={form.username} onChange={e => setForm({ ...form, username: e.target.value })}
                  placeholder="e.g. amen_employee1" autoFocus />
              </label>
            )}

            <label className="modal-field">
              <span>{form.mode === 'create' ? 'Password' : 'New password (leave blank to keep current)'}</span>
              <input type="password" value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
                placeholder="••••••••" autoComplete="new-password" />
            </label>

            <label className="modal-field">
              <span>Bank name</span>
              <input value={form.bank_name} onChange={e => setForm({ ...form, bank_name: e.target.value })}
                placeholder="e.g. Amen Bank" />
            </label>

            <label className="modal-field">
              <span>Role</span>
              <select value={form.role} onChange={e => setForm({ ...form, role: e.target.value })}>
                <option value="bank">Bank</option>
                <option value="admin">Administrator</option>
              </select>
            </label>

            {error && <div className="login-error">{error}</div>}

            <button type="submit" className="admin-primary-btn modal-submit" disabled={busy}>
              {busy ? <Loader2 size={16} className="spin" /> : null}
              {form.mode === 'create' ? 'Create account' : 'Save'}
            </button>
          </form>
        </div>
      )}
    </main>
  )
}
