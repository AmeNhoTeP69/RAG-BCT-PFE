import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../auth/AuthContext'
import { Plus, Pencil, Trash2, Check, X, Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react'

const EMPTY = { topic: '', correction: '' }
const FILTERS = [
  { key: '', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'approved', label: 'Approved' },
  { key: 'rejected', label: 'Rejected' },
]

export default function CorrectionsPanel() {
  const { authedFetch } = useAuth()
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [form, setForm] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const url = filter ? `/api/admin/corrections?status=${filter}` : '/api/admin/corrections'
      const res = await authedFetch(url)
      if (res.ok) setItems(await res.json())
    } finally { setLoading(false) }
  }, [authedFetch, filter])

  useEffect(() => { load() }, [load])

  const setStatus = async (c, status) => {
    await authedFetch(`/api/admin/corrections/${c.id}`, { method: 'PUT', body: JSON.stringify({ status }) })
    load()
  }
  const remove = async (c) => {
    if (!window.confirm('Permanently delete this correction?')) return
    await authedFetch(`/api/admin/corrections/${c.id}`, { method: 'DELETE' })
    load()
  }
  const openAdd = () => { setError(''); setForm({ mode: 'add', ...EMPTY }) }
  const openEdit = (c) => { setError(''); setForm({ mode: 'edit', id: c.id, topic: c.topic, correction: c.correction }) }

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      let res
      if (form.mode === 'add') {
        res = await authedFetch('/api/admin/corrections', {
          method: 'POST', body: JSON.stringify({ topic: form.topic, correction: form.correction }),
        })
      } else {
        res = await authedFetch(`/api/admin/corrections/${form.id}`, {
          method: 'PUT', body: JSON.stringify({ topic: form.topic, correction: form.correction }),
        })
      }
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || 'Something went wrong')
      setForm(null); load()
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  const badge = (s) =>
    s === 'approved' ? <span className="badge status-active"><CheckCircle2 size={11} /> Approved</span>
      : s === 'rejected' ? <span className="badge status-suspended"><XCircle size={11} /> Rejected</span>
        : <span className="badge status-pending"><Clock size={11} /> Pending</span>

  return (
    <main className="admin-page">
      <div className="admin-header">
        <div>
          <h1 className="admin-title">Verified corrections</h1>
          <p className="admin-subtitle">
            Review corrections captured from conversations. Once approved, they are injected as
            trusted notes into future answers (cross-session memory).
          </p>
        </div>
        <button className="admin-primary-btn" onClick={openAdd}><Plus size={16} /> Add a note</button>
      </div>

      <div className="filter-tabs">
        {FILTERS.map(f => (
          <button key={f.key} className={`filter-tab ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}>{f.label}</button>
        ))}
      </div>

      {loading ? (
        <div className="admin-empty"><Loader2 size={22} className="spin" /></div>
      ) : items.length === 0 ? (
        <div className="admin-empty">No corrections for this filter.</div>
      ) : (
        <div className="admin-table-wrap glass-card">
          <table className="admin-table">
            <thead>
              <tr><th>Topic</th><th>Correction (note)</th><th>Status</th><th>Created</th><th></th></tr>
            </thead>
            <tbody>
              {items.map(c => (
                <tr key={c.id}>
                  <td className="corr-topic">{c.topic}</td>
                  <td className="corr-note">{c.correction}</td>
                  <td>{badge(c.status)}</td>
                  <td className="muted-cell">{new Date(c.created_at).toLocaleDateString('en-GB')}</td>
                  <td>
                    <div className="row-actions">
                      {c.status !== 'approved' && (
                        <button onClick={() => setStatus(c, 'approved')} title="Approve" className="approve"><Check size={15} /></button>
                      )}
                      {c.status !== 'rejected' && (
                        <button onClick={() => setStatus(c, 'rejected')} title="Reject"><X size={15} /></button>
                      )}
                      <button onClick={() => openEdit(c)} title="Edit"><Pencil size={15} /></button>
                      <button onClick={() => remove(c)} className="danger" title="Delete"><Trash2 size={15} /></button>
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
              <h3>{form.mode === 'add' ? 'New verified note' : 'Edit correction'}</h3>
              <button type="button" onClick={() => setForm(null)}><X size={18} /></button>
            </div>
            <label className="modal-field">
              <span>Topic (used to match future questions semantically)</span>
              <input value={form.topic} onChange={e => setForm({ ...form, topic: e.target.value })}
                placeholder="e.g. foreign-exchange position monitoring — circular 2021-03 / 1997-08" autoFocus />
            </label>
            <label className="modal-field">
              <span>Correction / factual note</span>
              <textarea className="modal-textarea" value={form.correction}
                onChange={e => setForm({ ...form, correction: e.target.value })}
                placeholder="e.g. Circular 2021-03 (Art. 61) abrogates circular 1997-08…" rows={4} />
            </label>
            {error && <div className="login-error">{error}</div>}
            <button type="submit" className="admin-primary-btn modal-submit" disabled={busy}>
              {busy ? <Loader2 size={16} className="spin" /> : null} Save
            </button>
          </form>
        </div>
      )}
    </main>
  )
}
