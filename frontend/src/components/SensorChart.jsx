import React, { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

const CHANNEL_COLORS = {
  anomaly_score:      '#22d3ee',
  overall_risk_score: '#ef4444',
  bearing_temp_1:     '#a78bfa',
  bearing_temp_2:     '#818cf8',
  generator_temp:     '#f97316',
  gearbox_temp:       '#fbbf24',
  vibration_nacelle_x:'#34d399',
  vibration_nacelle_y:'#6ee7b7',
  rotor_speed:        '#60a5fa',
  power_output:       '#c084fc',
  oil_pressure:       '#fb7185',
  wind_speed:         '#4ade80',
}

const CHANNEL_LABELS = {
  anomaly_score:      'Anomaly',
  overall_risk_score: 'Risk',
  bearing_temp_1:     'Bearing T1',
  bearing_temp_2:     'Bearing T2',
  generator_temp:     'Gen Temp',
  gearbox_temp:       'Gearbox T',
  vibration_nacelle_x:'Vib X',
  vibration_nacelle_y:'Vib Y',
  rotor_speed:        'Rotor RPM',
  power_output:       'Power (kW)',
  oil_pressure:       'Oil Press',
  wind_speed:         'Wind (m/s)',
}

const DEFAULT_CHANNELS = ['anomaly_score', 'overall_risk_score', 'bearing_temp_1', 'generator_temp']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(6,11,24,0.95)',
      border: '1px solid rgba(34,211,238,0.2)',
      borderRadius: 8,
      padding: '8px 12px',
      fontSize: '0.78rem',
    }}>
      <div style={{ color: 'rgba(255,255,255,0.4)', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center gap-xs" style={{ marginBottom: 2 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: p.color, flexShrink: 0 }} />
          <span style={{ color: 'rgba(255,255,255,0.6)', marginRight: 4 }}>
            {CHANNEL_LABELS[p.dataKey] || p.dataKey}:
          </span>
          <span style={{ color: p.color, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * Live sensor chart — feeds from WebSocket stream or static chartData prop.
 * Props:
 *   chartData   : static array of reading objects (from /risk endpoint)
 *   liveStream  : array of live WebSocket readings appended in realtime
 *   channels    : array of field names to plot
 *   height      : chart height in px (default 220)
 *   turbineId   : for labelling
 */
export default function SensorChart({
  chartData = [],
  liveStream = [],
  channels = DEFAULT_CHANNELS,
  height = 220,
  turbineId,
}) {
  const [activeChannels, setActiveChannels] = useState(channels.slice(0, 4))
  const [view, setView] = useState('live')       // 'live' | 'history'

  const displayData = view === 'live' && liveStream.length > 0
    ? liveStream.slice(-60)
    : chartData.slice(-60)

  // Format timestamp for axis
  const formatted = displayData.map((d, i) => ({
    ...d,
    _t: d.timestamp
      ? new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : String(i),
  }))

  const toggleChannel = ch => {
    setActiveChannels(prev =>
      prev.includes(ch)
        ? prev.filter(c => c !== ch)
        : [...prev, ch]
    )
  }

  const availableChannels = channels.filter(c => c in CHANNEL_COLORS)

  return (
    <div className="card" style={{ padding: '16px' }}>
      {/* Header */}
      <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
        <div className="flex items-center gap-sm">
          {liveStream.length > 0 && view === 'live' && (
            <span className="live-dot" />
          )}
          <h4 style={{ color: 'var(--text-primary)', fontSize: '0.85rem' }}>
            {turbineId ? `${turbineId} — ` : ''}Sensor Monitor
          </h4>
        </div>
        <div className="flex gap-xs">
          <button
            className={`btn btn--ghost`}
            style={{
              padding: '4px 10px', fontSize: '0.72rem',
              ...(view === 'live' ? { color: 'var(--cyan)', borderColor: 'var(--border-bright)' } : {}),
            }}
            onClick={() => setView('live')}
          >Live</button>
          <button
            className="btn btn--ghost"
            style={{
              padding: '4px 10px', fontSize: '0.72rem',
              ...(view === 'history' ? { color: 'var(--cyan)', borderColor: 'var(--border-bright)' } : {}),
            }}
            onClick={() => setView('history')}
          >History</button>
        </div>
      </div>

      {/* Channel toggles */}
      <div className="flex wrap gap-xs" style={{ marginBottom: 12 }}>
        {availableChannels.map(ch => (
          <button
            key={ch}
            onClick={() => toggleChannel(ch)}
            style={{
              padding: '2px 8px',
              borderRadius: 100,
              fontSize: '0.68rem',
              fontWeight: 600,
              cursor: 'pointer',
              border: `1px solid ${activeChannels.includes(ch) ? CHANNEL_COLORS[ch] : 'rgba(255,255,255,0.1)'}`,
              background: activeChannels.includes(ch) ? `${CHANNEL_COLORS[ch]}18` : 'transparent',
              color: activeChannels.includes(ch) ? CHANNEL_COLORS[ch] : 'rgba(255,255,255,0.3)',
              transition: 'all 0.15s',
            }}
          >
            {CHANNEL_LABELS[ch] || ch}
          </button>
        ))}
      </div>

      {/* Chart */}
      {formatted.length === 0 ? (
        <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span className="text-muted text-sm">
            {view === 'live' ? 'Waiting for sensor stream…' : 'No history data'}
          </span>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={formatted} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="4 4" />
            <XAxis
              dataKey="_t"
              tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />
            {activeChannels.map(ch => (
              <Line
                key={ch}
                type="monotone"
                dataKey={ch}
                stroke={CHANNEL_COLORS[ch] || '#fff'}
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, strokeWidth: 0 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
