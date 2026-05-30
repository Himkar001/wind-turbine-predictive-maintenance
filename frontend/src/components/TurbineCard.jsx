import React from 'react'
import RiskGauge from './RiskGauge'

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

const URGENCY_CSS = { critical: 'critical', high: 'high', medium: 'medium', low: 'low' }

function MetricRow({ label, value, unit = '', highlight }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '7px 0',
      borderBottom: '1px solid rgba(255,255,255,0.04)',
    }}>
      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </span>
      <span style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.85rem',
        fontWeight: 700,
        color: highlight || 'var(--text-primary)',
      }}>
        {value}{unit}
      </span>
    </div>
  )
}

/**
 * Detailed per-turbine info card.
 * Shows gauge, fault, RUL, anomaly, confidence, dominant signal.
 */
export default function TurbineCard({ data, onAnalyze, analyzing }) {
  if (!data) return null

  const tier     = (data.risk_tier || 'LOW').toUpperCase()
  const css      = TIER_CSS[tier] || 'low'
  const fault    = data.fault_type || 'unknown'
  const urgency  = (data.rul_urgency || 'low').toLowerCase()
  const urgCss   = URGENCY_CSS[urgency] || 'low'
  const signal   = data.dominant_signal || 'anomaly'

  return (
    <div className={`card card--${css}`} style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {/* Turbine header */}
      <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', color: 'var(--text-primary)' }}>
            {data.turbine_id}
          </h2>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
            Last reading: {data.timestamp
              ? new Date(data.timestamp).toLocaleString()
              : '—'}
          </div>
        </div>
        <span className={`badge badge--${css}`} style={{ fontSize: '0.78rem' }}>{tier}</span>
      </div>

      {/* Gauge */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
        <RiskGauge
          score={data.risk_score || 0}
          tier={tier}
          turbineId={data.turbine_id}
          size={180}
        />
      </div>

      {/* Fault banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 14px',
        borderRadius: 8,
        background: `var(--${css}-glow)`,
        border: `1px solid rgba(var(--${css}),0.2)`,
        marginBottom: 14,
      }}>
        <span style={{ fontSize: '1.4rem' }}>{FAULT_ICONS[fault] || '❓'}</span>
        <div>
          <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'capitalize' }}>
            {fault.replace(/_/g, ' ')}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Confidence: {(data.fault_confidence * 100).toFixed(0)}% · Driven by: {signal}
          </div>
        </div>
      </div>

      {/* Metrics */}
      <MetricRow label="Anomaly Score"    value={(data.anomaly_score * 100).toFixed(1)} unit="%" highlight="var(--cyan)" />
      <MetricRow
        label="Remaining Life"
        value={typeof data.rul_days === 'number' ? data.rul_days.toFixed(1) : '—'}
        unit=" days"
        highlight={`var(--${urgCss})`}
      />
      <MetricRow
        label="RUL Urgency"
        value={urgency.toUpperCase()}
        highlight={`var(--${urgCss})`}
      />
      <MetricRow label="Fault Confidence" value={(data.fault_confidence * 100).toFixed(1)} unit="%" />

      {/* Risk breakdown */}
      {data.risk_breakdown && Object.keys(data.risk_breakdown).length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div className="section-title" style={{ marginBottom: 8 }}>Risk Breakdown</div>
          {['anomaly_contribution', 'fault_contribution', 'rul_contribution'].map(k => {
            const val  = data.risk_breakdown[k] ?? 0
            const pct  = Math.min(100, Math.round(val * 100 / 0.5))
            const label = k.replace('_contribution', '').toUpperCase()
            return (
              <div key={k} style={{ marginBottom: 6 }}>
                <div className="flex justify-between" style={{ marginBottom: 3 }}>
                  <span className="text-xs text-muted">{label}</span>
                  <span className="text-xs mono-num text-cyan">{val.toFixed(3)}</span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)' }}>
                  <div style={{
                    width: `${pct}%`, height: '100%', borderRadius: 2,
                    background: 'var(--cyan)', opacity: 0.7,
                  }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Analyze button */}
      <button
        id={`analyze-btn-${data.turbine_id}`}
        className="btn btn--primary"
        onClick={() => onAnalyze && onAnalyze(data.turbine_id)}
        disabled={analyzing}
        style={{ marginTop: 14, width: '100%', justifyContent: 'center' }}
      >
        {analyzing ? (
          <><span className="animate-spin" style={{ display: 'inline-block' }}>⟳</span> Running Pipeline…</>
        ) : (
          '🤖 Run AI Analysis'
        )}
      </button>
    </div>
  )
}
