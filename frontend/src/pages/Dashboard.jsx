import React, { useState, useEffect, useCallback, useRef } from 'react'
import FleetMap     from '../components/FleetMap'
import TurbineCard  from '../components/TurbineCard'
import SensorChart  from '../components/SensorChart'
import AlertFeed    from '../components/AlertFeed'
import ChatBot      from '../components/ChatBot'
import {
  fetchFleetStatus, fetchFleetSummary,
  fetchRisk, fetchAlerts,
  postAnalyze,
  subscribeAlerts, subscribeSensors,
} from '../api/client'

// ── KPI summary bar ───────────────────────────────────────
function KpiBar({ summary, wsStatus }) {
  if (!summary) return null
  const { total_turbines, tier_breakdown = {}, average_risk, most_at_risk } = summary

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
      gap: 12,
      marginBottom: 'var(--gap-lg)',
    }}>
      {/* Total */}
      <div className="card" style={{ padding: '14px 18px' }}>
        <div className="stat__label">Total Turbines</div>
        <div className="stat__value">{total_turbines}</div>
        <div className="stat__sub">in fleet</div>
      </div>

      {/* Tier counts */}
      {[
        { key: 'CRITICAL', label: 'Critical', var: '--critical' },
        { key: 'HIGH',     label: 'High Risk', var: '--high' },
        { key: 'MEDIUM',   label: 'Medium',   var: '--medium' },
        { key: 'LOW',      label: 'Healthy',  var: '--low' },
      ].map(({ key, label, var: cssVar }) => (
        <div key={key} className="card" style={{ padding: '14px 18px', borderColor: `${cssVar.replace('--', 'rgba(var(')},.2)` }}>
          <div className="stat__label">{label}</div>
          <div className="stat__value" style={{ color: `var(${cssVar})` }}>
            {tier_breakdown[key] ?? 0}
          </div>
          <div className="stat__sub">turbines</div>
        </div>
      ))}

      {/* Avg risk */}
      <div className="card" style={{ padding: '14px 18px' }}>
        <div className="stat__label">Avg Risk</div>
        <div className="stat__value text-cyan">{((average_risk || 0) * 100).toFixed(1)}%</div>
        <div className="stat__sub">fleet-wide</div>
      </div>

      {/* WebSocket status */}
      <div className="card" style={{ padding: '14px 18px' }}>
        <div className="stat__label">Stream</div>
        <div className="flex items-center gap-sm" style={{ marginTop: 4 }}>
          <span className={`live-dot ${wsStatus !== 'connected' ? 'live-dot--medium' : ''}`} />
          <span className="stat__value" style={{ fontSize: '1rem' }}>
            {wsStatus === 'connected' ? 'Live' : wsStatus === 'connecting' ? 'Connecting' : 'Offline'}
          </span>
        </div>
      </div>
    </div>
  )
}


// ── Main Dashboard ────────────────────────────────────────
export default function Dashboard() {
  const [fleet,       setFleet]       = useState([])
  const [summary,     setSummary]     = useState(null)
  const [alerts,      setAlerts]      = useState([])
  const [selectedId,  setSelectedId]  = useState(null)
  const [turbineData, setTurbineData] = useState(null)
  const [liveStream,  setLiveStream]  = useState([])
  const [wsStatus,    setWsStatus]    = useState('connecting')
  const [analyzing,   setAnalyzing]   = useState(false)
  const [chatOpen,    setChatOpen]    = useState(false)
  const [loadingAlerts, setLoadingAlerts] = useState(true)
  const sensorUnsub = useRef(null)

  // ── Initial data fetch ────────────────────────────────────
  useEffect(() => {
    fetchFleetStatus().then(r => setFleet(r.data.data || [])).catch(console.warn)
    fetchFleetSummary().then(r => setSummary(r.data)).catch(console.warn)

    // Fetch alerts with retry — backend may take up to 60s to warm up parquet
    const loadAlerts = (attempt = 0) => {
      fetchAlerts({ limit: 100, min_tier: 'MEDIUM' })
        .then(r => {
          const list = r.data.alerts || []
          setAlerts(list)
          setLoadingAlerts(false)
          // Retry if empty and we haven't tried too many times
          if (list.length === 0 && attempt < 6) {
            setTimeout(() => loadAlerts(attempt + 1), 8000)
          }
        })
        .catch(() => {
          if (attempt < 6) setTimeout(() => loadAlerts(attempt + 1), 8000)
          else setLoadingAlerts(false)
        })
    }
    loadAlerts()
  }, [])

  // ── Refresh fleet + alerts every 60 s ─────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      fetchFleetStatus().then(r => setFleet(r.data.data || [])).catch(() => {})
      fetchFleetSummary().then(r => setSummary(r.data)).catch(() => {})
      fetchAlerts({ limit: 100, min_tier: 'MEDIUM' })
        .then(r => setAlerts(r.data.alerts || []))
        .catch(() => {})
    }, 60_000)
    return () => clearInterval(id)
  }, [])


  // ── Alert WebSocket ───────────────────────────────────────
  useEffect(() => {
    setWsStatus('connecting')
    const unsub = subscribeAlerts(msg => {
      setWsStatus('connected')
      if (msg.event === 'fleet_heartbeat') {
        // nothing extra needed
      } else if (msg.event === 'analysis_complete') {
        // Refresh alerts
        fetchAlerts({ limit: 100, min_tier: 'MEDIUM' })
          .then(r => setAlerts(r.data.alerts || []))
          .catch(() => {})
      }
    }, () => setWsStatus('error'))
    return unsub
  }, [])

  // ── Sensor WebSocket (per turbine) ────────────────────────
  const connectSensorStream = useCallback((turbineId) => {
    if (sensorUnsub.current) {
      sensorUnsub.current()
      sensorUnsub.current = null
    }
    setLiveStream([])
    if (!turbineId) return

    sensorUnsub.current = subscribeSensors(turbineId, msg => {
      if (msg.event === 'sensor_reading') {
        setLiveStream(prev => [...prev.slice(-120), msg])
      }
    })
  }, [])

  // ── Select turbine ────────────────────────────────────────
  const handleSelect = useCallback(async (turbineId) => {
    setSelectedId(turbineId)
    connectSensorStream(turbineId)
    try {
      const r = await fetchRisk(turbineId, 50)
      setTurbineData(r.data)
    } catch (err) {
      console.warn('Failed to fetch risk for', turbineId, err)
    }
  }, [connectSensorStream])

  // ── AI Analysis ───────────────────────────────────────────
  const handleAnalyze = useCallback(async (turbineId) => {
    setAnalyzing(true)
    try {
      await postAnalyze(turbineId, null)
      const r = await fetchRisk(turbineId, 50)
      setTurbineData(r.data)
    } catch (err) {
      console.warn('Analysis failed:', err)
    } finally {
      setAnalyzing(false)
    }
  }, [])

  // Cleanup sensor WS on unmount
  useEffect(() => () => sensorUnsub.current?.(), [])

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Navbar ─────────────────────────────────────────── */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        background: 'rgba(6,11,24,0.9)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{
          maxWidth: 1600, margin: '0 auto',
          padding: '14px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div className="flex items-center gap-md">
            <span style={{ fontSize: '1.5rem' }}>🌬️</span>
            <div>
              <h1 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                WindSense <span className="text-cyan">AI</span>
              </h1>
              <div className="text-xs text-muted font-medium tracking-wide mt-1">
                Predictive Maintenance
              </div>
            </div>
          </div>

          <div className="flex items-center gap-md">
            <div className="flex items-center gap-xs">
              <span className={`live-dot ${wsStatus !== 'connected' ? 'live-dot--medium' : ''}`} />
              <span className="text-xs text-sec">{wsStatus}</span>
            </div>
            <button
              id="chat-toggle"
              className="btn btn--ghost"
              onClick={() => setChatOpen(o => !o)}
              style={{ gap: 6 }}
            >
              🤖 WindSense AI
            </button>
          </div>
        </div>
      </header>

      {/* ── Main content ───────────────────────────────────── */}
      <main style={{ flex: 1, maxWidth: 1600, margin: '0 auto', width: '100%', padding: '24px' }}>

        {/* KPI bar */}
        <KpiBar summary={summary} wsStatus={wsStatus} />

        {/* Two-column layout */}
        <div style={{ display: 'grid', gridTemplateColumns: selectedId ? '1fr 340px' : '1fr', gap: 20 }}>

          {/* Left column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
            {/* Fleet map */}
            <div className="card">
              <FleetMap
                turbines={fleet}
                onSelect={handleSelect}
                selectedId={selectedId}
              />
            </div>

            {/* Sensor chart */}
            {selectedId && (
              <SensorChart
                chartData={turbineData?.chart_data || []}
                liveStream={liveStream}
                turbineId={selectedId}
                height={240}
                channels={[
                  'anomaly_score', 'overall_risk_score',
                  'bearing_temp_1', 'generator_temp',
                  'vibration_nacelle_x', 'power_output',
                  'rotor_speed', 'wind_speed',
                ]}
              />
            )}

            {/* Alert feed */}
            <AlertFeed alerts={alerts} loading={loadingAlerts} maxHeight={360} />
          </div>

          {/* Right panel — turbine detail */}
          {selectedId && (
            <div style={{ position: 'sticky', top: 80, alignSelf: 'start' }}>
              <TurbineCard
                data={turbineData}
                onAnalyze={handleAnalyze}
                analyzing={analyzing}
              />
            </div>
          )}
        </div>
      </main>

      {/* ── Floating chatbot ───────────────────────────────── */}
      <ChatBot
        turbineId={selectedId}
        turbineData={turbineData}
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
      />

      {/* Chat FAB (visible when chat is closed) */}
      {!chatOpen && (
        <button
          id="chat-fab"
          onClick={() => setChatOpen(true)}
          style={{
            position: 'fixed',
            bottom: 20,
            right: 20,
            width: 52,
            height: 52,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #7c3aed, #a78bfa)',
            border: 'none',
            cursor: 'pointer',
            fontSize: '1.3rem',
            boxShadow: '0 4px 20px rgba(124,58,237,0.4)',
            zIndex: 999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'transform 0.2s',
          }}
          onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
        >
          🤖
        </button>
      )}
    </div>
  )
}
