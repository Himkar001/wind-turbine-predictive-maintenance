import React, { useMemo } from 'react'

const TIER_RANK = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 }

const TIER_CSS = {
  CRITICAL: 'critical',
  HIGH:     'high',
  MEDIUM:   'medium',
  LOW:      'low',
}

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

function TurbineCard({ turbine, onSelect, selected }) {
  const tier  = (turbine.risk_tier || turbine.tier || 'LOW').toUpperCase()
  const css   = TIER_CSS[tier] || 'low'
  const score = turbine.p95_risk_score ?? turbine.mean_risk_score ?? turbine.risk_score ?? 0
  const fault = turbine.fault_type || turbine.dominant_fault || turbine.predicted_fault || 'normal'
  const rul   = turbine.min_rul_days ?? turbine.rul_days ?? '--'

  const pct = Math.min(100, Math.round(score * 100))

  return (
    <button
      id={`turbine-card-${turbine.turbine_id}`}
      className={`card card--${css} turbine-card ${selected ? 'turbine-card--selected' : ''}`}
      onClick={() => onSelect(turbine.turbine_id)}
      style={{ cursor: 'pointer', textAlign: 'left', width: '100%' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
        <div className="flex items-center gap-sm">
          <span className={`live-dot live-dot--${css}`} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            {turbine.turbine_id}
          </span>
        </div>
        <span className={`badge badge--${css}`}>{tier}</span>
      </div>

      {/* Risk bar */}
      <div style={{ marginBottom: 12 }}>
        <div className="flex justify-between" style={{ marginBottom: 4 }}>
          <span className="text-xs text-muted">RISK SCORE</span>
          <span className={`text-xs mono-num risk-${css.toLowerCase()}`}>{(score * 100).toFixed(1)}%</span>
        </div>
        <div style={{
          height: 6,
          borderRadius: 3,
          background: 'var(--bg-elevated)',
          overflow: 'hidden',
        }}>
          <div style={{
            width: `${pct}%`,
            height: '100%',
            borderRadius: 3,
            background: `var(--${css})`,
            boxShadow: `0 0 8px var(--${css}-glow)`,
            transition: 'width 0.6s ease',
          }} />
        </div>
      </div>

      {/* Stats */}
      <div className="grid-2" style={{ gap: 8 }}>
        <div className="stat">
          <span className="stat__label">Fault</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600 }}>
            {FAULT_ICONS[fault] || '❓'} {fault.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="stat">
          <span className="stat__label">RUL</span>
          <span style={{ fontSize: '0.9rem', fontFamily: 'var(--font-mono)', color: 'var(--amber)', fontWeight: 700 }}>
            {typeof rul === 'number' ? `${rul.toFixed(1)}d` : rul}
          </span>
        </div>
      </div>
    </button>
  )
}

export default function FleetMap({ turbines = [], onSelect, selectedId }) {
  const sorted = useMemo(() =>
    [...turbines].sort((a, b) =>
      (TIER_RANK[(b.risk_tier || 'LOW').toUpperCase()] || 0) -
      (TIER_RANK[(a.risk_tier || 'LOW').toUpperCase()] || 0)
    ),
    [turbines]
  )

  if (!turbines.length) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
        <div className="skeleton" style={{ height: 20, width: '60%', margin: '0 auto 12px' }} />
        <div className="skeleton" style={{ height: 12, width: '40%', margin: '0 auto' }} />
        <p style={{ marginTop: 16 }} className="text-muted text-sm">Loading fleet data…</p>
      </div>
    )
  }

  return (
    <div>
      <div className="section-header">
        <span className="section-title">Wind Farm — {turbines.length} Turbines</span>
        <span className="badge badge--online">
          <span className="live-dot" style={{ width: 6, height: 6 }} />
          Live
        </span>
      </div>
      <div className="grid-3" style={{ gap: 12 }}>
        {sorted.map(t => (
          <TurbineCard
            key={t.turbine_id}
            turbine={t}
            onSelect={onSelect}
            selected={t.turbine_id === selectedId}
          />
        ))}
      </div>
    </div>
  )
}
