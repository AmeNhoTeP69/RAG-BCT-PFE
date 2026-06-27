import { useState, useEffect } from 'react'
import { useAuth } from '../auth/AuthContext'
import { Line, Doughnut, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Tooltip, Legend, Filler,
} from 'chart.js'
import { MessageSquareText, Users, Clock, Zap, Loader2 } from 'lucide-react'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Tooltip, Legend, Filler,
)

const GOLD = '#D4AF37'
const BLUE = '#4f8ef7'
const TEXT = '#b0b4c8'
const GRID = 'rgba(255,255,255,0.06)'
const TOPIC_COLORS = ['#D4AF37', '#4f8ef7', '#e0bd5a', '#6ba1f9', '#b8962e', '#3b6fd4', '#f0d27e', '#8bb6fb']

const shortDate = (iso) => `${iso.slice(8, 10)}/${iso.slice(5, 7)}`

const axisOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: TEXT, font: { size: 10 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 10 }, grid: { color: GRID } },
    y: { ticks: { color: TEXT, font: { size: 10 } }, grid: { color: GRID }, beginAtZero: true },
  },
}

function KpiCard({ icon, value, label }) {
  return (
    <div className="kpi-card glass-card">
      <div className="kpi-icon">{icon}</div>
      <div>
        <div className="kpi-value">{value}</div>
        <div className="kpi-label">{label}</div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { authedFetch } = useAuth()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    authedFetch('/api/admin/analytics?days=30')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (!cancelled) { setData(d); setLoading(false) } })
      .catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [authedFetch])

  if (loading) return <main className="admin-page"><div className="admin-empty"><Loader2 size={22} className="spin" /></div></main>
  if (!data) return <main className="admin-page"><div className="admin-empty">Data unavailable.</div></main>

  const k = data.kpis
  const rt = k.avg_response_time_ms >= 1000 ? `${(k.avg_response_time_ms / 1000).toFixed(1)} s` : `${k.avg_response_time_ms} ms`
  const noData = k.total_questions === 0

  // Chart 1 — questions per day (area)
  const qpd = {
    labels: data.questions_per_day.map(d => shortDate(d.date)),
    datasets: [{
      label: 'Questions', data: data.questions_per_day.map(d => d.count),
      borderColor: GOLD, backgroundColor: 'rgba(212,175,55,0.15)',
      fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2,
    }],
  }

  // Chart 2 — topic distribution (doughnut)
  const topics = {
    labels: data.topics.map(t => t.label),
    datasets: [{
      data: data.topics.map(t => t.count),
      backgroundColor: TOPIC_COLORS, borderColor: '#0d1120', borderWidth: 2,
    }],
  }

  // Chart 3 — RAG vs Graph (bar)
  const modes = {
    labels: ['Standard RAG', 'Graph RAG'],
    datasets: [{
      label: 'Queries', data: [data.modes.rag, data.modes.graph],
      backgroundColor: [GOLD, BLUE], borderRadius: 8, barThickness: 64,
    }],
  }

  // Chart 4 — top banks/users (horizontal bar)
  const top = {
    labels: data.top_users.map(u => u.label),
    datasets: [{
      label: 'Queries', data: data.top_users.map(u => u.count),
      backgroundColor: 'rgba(79,142,247,0.55)', borderColor: BLUE, borderWidth: 1, borderRadius: 6,
    }],
  }

  // Chart 5 — cache hit rate over time (area)
  const cache = {
    labels: data.cache_rate_per_day.map(d => shortDate(d.date)),
    datasets: [{
      label: 'Cache rate', data: data.cache_rate_per_day.map(d => Math.round(d.rate * 100)),
      borderColor: BLUE, backgroundColor: 'rgba(79,142,247,0.15)',
      fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2,
    }],
  }

  return (
    <main className="admin-page">
      <div className="admin-header">
        <div>
          <h1 className="admin-title">Analytics dashboard</h1>
          <p className="admin-subtitle">Real system usage over the last 30 days.</p>
        </div>
      </div>

      {noData && <div className="dashboard-note">No queries recorded yet — the charts will fill in once questions are asked.</div>}

      <div className="kpi-row">
        <KpiCard icon={<MessageSquareText size={20} />} value={k.total_questions} label="Questions asked" />
        <KpiCard icon={<Users size={20} />} value={k.active_users_7d} label="Active users (7d)" />
        <KpiCard icon={<Clock size={20} />} value={rt} label="Avg. response time" />
        <KpiCard icon={<Zap size={20} />} value={`${Math.round(k.cache_hit_rate * 100)}%`} label="Cache hit rate" />
      </div>

      <div className="dashboard-grid">
        <div className="chart-card glass-card chart-wide">
          <div className="chart-card-title">Questions per day</div>
          <div className="chart-canvas"><Line data={qpd} options={axisOpts} /></div>
        </div>

        <div className="chart-card glass-card">
          <div className="chart-card-title">Topic distribution (LDA)</div>
          <div className="chart-canvas">
            <Doughnut data={topics} options={{
              responsive: true, maintainAspectRatio: false, cutout: '58%',
              plugins: { legend: { position: 'bottom', labels: { color: TEXT, font: { size: 9.5 }, boxWidth: 10, padding: 8 } } },
            }} />
          </div>
        </div>

        <div className="chart-card glass-card">
          <div className="chart-card-title">Standard RAG vs Graph RAG</div>
          <div className="chart-canvas"><Bar data={modes} options={axisOpts} /></div>
        </div>

        <div className="chart-card glass-card">
          <div className="chart-card-title">Most active banks</div>
          <div className="chart-canvas">
            <Bar data={top} options={{ ...axisOpts, indexAxis: 'y' }} />
          </div>
        </div>

        <div className="chart-card glass-card">
          <div className="chart-card-title">Cache hit rate over time (%)</div>
          <div className="chart-canvas">
            <Line data={cache} options={{ ...axisOpts, scales: { ...axisOpts.scales, y: { ...axisOpts.scales.y, max: 100 } } }} />
          </div>
        </div>
      </div>
    </main>
  )
}
