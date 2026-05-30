import React, { useEffect, useRef } from 'react'

const TIER_CSS = { CRITICAL: 'critical', HIGH: 'high', MEDIUM: 'medium', LOW: 'low' }

const FAULT_ICONS = {
  bearing_failure:       '⚙️',
  gearbox_fault:         '🔩',
  generator_overheating: '🌡️',
  electrical_fault:      '⚡',
  oil_leak:              '🛢️',
  blade_imbalance:       '🌀',
  maintenance:           '🔧',
  normal:                '✅',
}

function AlertRow({ alert, index }) {
  const tier    = (alert.risk_tier || 'LOW').toUpperCase()
  const css     = TIER_CSS[tier] || 'low'
  const fault   = alert.fault_type || 'unknown'
  const ts      = alert.timestamp
    ? new Date(alert.timestamp).toLocaleString([], {
        month: 'short', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      })
    : '—'

  return (
    <div
      className="animate-fade-in"
      style={{
        display: 'grid',
        gridTemplateColumns: '80px 1fr 100px 80px 80px',
        alignItems: 'center',
        gap: 8,
        padding: '10px 12px',
        borderRadius: 8,
        background: index % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent',
        borderLeft: `3px solid var(--${css})`,
        animationDelay: `${Math.min(index, 10) * 40}ms`,
      }}
    >
      {/* Turbine ID */}
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
        {alert.turbine_id}
      </span>

      {/* Fault + timestamp */}
      <div>
        <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 500 }}>
          {FAULT_ICONS[fault] || '❓'} {fault.replace(/_/g, ' ')}
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 1 }}>{ts}</div>
      </div>

      {/* Risk score */}
      <div style={{ textAlign: 'right' }}>
        <span className={`mono-num risk-${css}`} style={{ fontSize: '0.9rem', fontWeight: 700 }}>
          {(alert.risk_score * 100).toFixed(1)}%
        </span>
        <span className={`badge badge--${css}`} style={{ display: 'block', marginTop: 3 }}>
          {tier}
        </span>
      </div>

      {/* RUL */}
      <div style={{ textAlign: 'right' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--amber)', fontWeight: 600 }}>
          {typeof alert.rul_days === 'number' ? `${alert.rul_days.toFixed(1)}d` : '—'}
        </span>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2, textTransform: 'uppercase' }}>
          {alert.rul_urgency || '—'}
        </div>
      </div>

      {/* Anomaly */}
      <div style={{ textAlign: 'right' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--cyan)' }}>
          {typeof alert.anomaly_score === 'number' ? alert.anomaly_score.toFixed(3) : '—'}
        </span>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>anomaly</div>
      </div>
    </div>
  )
}

export default function AlertFeed({ alerts = [], loading = false, maxHeight = 400 }) {
  const bottomRef = useRef(null)

  // Auto-scroll when new alerts arrive
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [alerts.length])

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        padding: '14px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div className="flex items-center gap-sm">
          <span className="live-dot" />
          <h3 style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>Alert Feed</h3>
        </div>
        <span className="text-muted text-xs">{alerts.length} alerts</span>
      </div>

      {/* Column headers */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '80px 1fr 100px 80px 80px',
        gap: 8,
        padding: '6px 12px',
        borderBottom: '1px solid var(--border)',
      }}>
        {['Turbine', 'Fault / Time', 'Risk', 'RUL', 'Anomaly'].map(h => (
          <span key={h} className="section-title" style={{ fontSize: '0.65rem' }}>{h}</span>
        ))}
      </div>

      {/* Rows */}
      <div
        id="alert-feed-list"
        className="scroll-list"
        style={{ maxHeight, overflowY: 'auto' }}
      >
        {loading && alerts.length === 0 ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} style={{ padding: '10px 12px' }}>
              <div className="skeleton" style={{ height: 16, borderRadius: 4 }} />
            </div>
          ))
        ) : alerts.length === 0 ? (
          <div style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            No alerts matching current filter
          </div>
        ) : (
          alerts.map((alert, i) => (
            <AlertRow key={`${alert.turbine_id}-${alert.timestamp}-${i}`} alert={alert} index={i} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
