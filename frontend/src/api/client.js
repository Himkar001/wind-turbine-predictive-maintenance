import axios from 'axios'

// ── Base Axios client ─────────────────────────────────────
const BASE = import.meta.env.VITE_API_URL || '/api'

export const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Endpoints ─────────────────────────────────────────────
export const fetchHealth        = ()        => api.get('/health')
export const fetchFleetStatus   = ()        => api.get('/fleet/status')
export const fetchFleetSummary  = ()        => api.get('/fleet/summary')
export const fetchTurbines      = ()        => api.get('/turbines')
export const fetchRisk          = (id, n=50)=> api.get(`/risk/${id}?last_n=${n}`)
export const fetchAlerts        = (params)  => api.get('/alerts', { params })
export const postPredict        = (body)    => api.post('/predict', body)
export const postAnalyze        = (id, body)=> api.post(`/analyze/${id}`, body)
export const postChat           = (body)    => api.post('/chat', body)
export const fetchReport        = (id)      => api.get(`/report/${id}`)
export const fetchReports       = (params)  => api.get('/reports/list', { params })

// ── WebSocket helpers ─────────────────────────────────────
const WS_BASE = (() => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host  = window.location.host
  return `${proto}://${host}/ws`
})()

/**
 * Subscribe to the fleet alert broadcast stream.
 * Returns a cleanup function to close the socket.
 */
export function subscribeAlerts(onMessage, onError) {
  const ws = new WebSocket(`${WS_BASE}/alerts`)
  ws.onmessage = e => {
    try { onMessage(JSON.parse(e.data)) }
    catch (_) {}
  }
  ws.onerror = onError || (() => {})
  return () => ws.readyState < 2 && ws.close()
}

/**
 * Subscribe to the per-turbine sensor stream.
 * Returns a cleanup function.
 */
export function subscribeSensors(turbineId, onMessage, onError) {
  const ws = new WebSocket(`${WS_BASE}/sensors/${turbineId}`)
  ws.onmessage = e => {
    try { onMessage(JSON.parse(e.data)) }
    catch (_) {}
  }
  ws.onerror = onError || (() => {})
  return () => ws.readyState < 2 && ws.close()
}
