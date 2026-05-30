import React, { useMemo } from 'react'

const TIER_CSS = { CRITICAL: 'critical', HIGH: 'high', MEDIUM: 'medium', LOW: 'low' }

/**
 * Semi-circle SVG gauge for risk score.
 * score: 0..1, tier: CRITICAL|HIGH|MEDIUM|LOW
 */
export default function RiskGauge({ score = 0, tier = 'LOW', turbineId, size = 160 }) {
  const tierKey = (tier || 'LOW').toUpperCase()
  const css     = TIER_CSS[tierKey] || 'low'

  const { strokeColor, glowColor } = useMemo(() => ({
    CRITICAL: { strokeColor: '#ef4444', glowColor: 'rgba(239,68,68,0.4)' },
    HIGH:     { strokeColor: '#f97316', glowColor: 'rgba(249,115,22,0.4)' },
    MEDIUM:   { strokeColor: '#eab308', glowColor: 'rgba(234,179,8,0.4)'  },
    LOW:      { strokeColor: '#22c55e', glowColor: 'rgba(34,197,94,0.4)'  },
  }[tierKey] || { strokeColor: '#22c55e', glowColor: 'rgba(34,197,94,0.4)' }), [tierKey])

  // SVG arc geometry
  const cx = size / 2
  const cy = size * 0.58
  const r  = size * 0.38
  const strokeW = size * 0.07

  // Arc from 180° to 0° (left to right, semi-circle above center line)
  const startAngle = Math.PI          // left point
  const endAngle   = 0               // right point
  const clampedScore = Math.max(0, Math.min(1, score))
  const sweepAngle = startAngle - (startAngle - endAngle) * clampedScore

  const arcX = cx + r * Math.cos(sweepAngle)
  const arcY = cy - r * Math.sin(sweepAngle)  // SVG y is inverted

  const trackStart  = { x: cx - r, y: cy }
  const trackEnd    = { x: cx + r, y: cy }

  // Active arc path (semi-circle from left up to current score)
  // Since it's a semi-circle (max 180 degrees), the large-arc flag is always 0.
  const arcPath = clampedScore > 0
    ? `M ${trackStart.x} ${trackStart.y} A ${r} ${r} 0 0 1 ${arcX} ${arcY}`
    : ''

  // Track (full semi-circle)
  const trackPath = `M ${trackStart.x} ${trackStart.y} A ${r} ${r} 0 0 1 ${trackEnd.x} ${trackEnd.y}`


  const pct = Math.round(clampedScore * 100)

  return (
    <div
      id={`risk-gauge-${turbineId || 'gauge'}`}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}
    >
      <svg
        width={size}
        height={size * 0.65}
        viewBox={`0 0 ${size} ${size * 0.65}`}
        style={{ overflow: 'visible' }}
      >
        <defs>
          <filter id={`glow-${turbineId}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background track */}
        <path
          d={trackPath}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />

        {/* Active arc */}
        {arcPath && (
          <path
            d={arcPath}
            fill="none"
            stroke={strokeColor}
            strokeWidth={strokeW}
            strokeLinecap="round"
            filter={`url(#glow-${turbineId})`}
            style={{ transition: 'all 0.8s cubic-bezier(0.4,0,0.2,1)' }}
          />
        )}

        {/* Center score text */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          fill={strokeColor}
          fontFamily="'JetBrains Mono', monospace"
          fontSize={size * 0.155}
          fontWeight="700"
        >
          {pct}%
        </text>
        <text
          x={cx}
          y={cy + size * 0.1}
          textAnchor="middle"
          fill="rgba(255,255,255,0.3)"
          fontFamily="'Inter', sans-serif"
          fontSize={size * 0.07}
          fontWeight="600"
          letterSpacing="2"
          textTransform="uppercase"
        >
          RISK
        </text>

        {/* Min/Max labels */}
        <text x={trackStart.x - 2} y={cy + 14} textAnchor="middle" fill="rgba(255,255,255,0.2)" fontSize={size * 0.065}>0</text>
        <text x={trackEnd.x + 2}   y={cy + 14} textAnchor="middle" fill="rgba(255,255,255,0.2)" fontSize={size * 0.065}>100</text>
      </svg>

      {/* Tier badge */}
      <span className={`badge badge--${css}`} style={{ marginTop: -4 }}>
        {tierKey}
      </span>
    </div>
  )
}
