import React, { useState, useRef, useEffect } from 'react'
import { postChat } from '../api/client'

const SUGGESTED = [
  'What does the current risk score mean?',
  'When should I schedule maintenance?',
  'Explain the bearing failure fault',
  'Which turbine needs attention first?',
]

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div
      className="animate-fade-in"
      style={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        gap: 8,
        marginBottom: 12,
        alignItems: 'flex-start',
      }}
    >
      {/* Avatar */}
      <div style={{
        width: 28,
        height: 28,
        borderRadius: '50%',
        background: isUser
          ? 'linear-gradient(135deg, #0891b2, #22d3ee)'
          : 'linear-gradient(135deg, #7c3aed, #a78bfa)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.7rem',
        fontWeight: 700,
        color: '#fff',
        flexShrink: 0,
      }}>
        {isUser ? 'U' : '🤖'}
      </div>

      {/* Bubble */}
      <div style={{
        maxWidth: '80%',
        background: isUser
          ? 'linear-gradient(135deg, rgba(8,145,178,0.2), rgba(34,211,238,0.1))'
          : 'rgba(255,255,255,0.04)',
        border: `1px solid ${isUser ? 'rgba(34,211,238,0.2)' : 'rgba(255,255,255,0.06)'}`,
        borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
        padding: '10px 14px',
        fontSize: '0.85rem',
        lineHeight: 1.6,
        color: isUser ? 'var(--text-primary)' : 'var(--text-secondary)',
        whiteSpace: 'pre-wrap',
      }}>
        {msg.content}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 12 }}>
      <div style={{
        width: 28, height: 28, borderRadius: '50%',
        background: 'linear-gradient(135deg, #7c3aed, #a78bfa)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.7rem', flexShrink: 0,
      }}>🤖</div>
      <div style={{
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '12px 12px 12px 4px',
        padding: '12px 16px',
        display: 'flex', gap: 4, alignItems: 'center',
      }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 6, height: 6, borderRadius: '50%', background: 'var(--cyan)',
            animation: 'pulse-dot 1.2s ease-in-out infinite',
            animationDelay: `${i * 0.2}s`,
          }} />
        ))}
      </div>
    </div>
  )
}

/**
 * Floating chatbot panel.
 * Props:
 *   turbineId   : current selected turbine (sent as context)
 *   turbineData : current turbine risk data dict (for enriching chat)
 *   isOpen      : boolean
 *   onClose     : callback
 */
export default function ChatBot({ turbineId, turbineData, isOpen, onClose }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hi! I\'m WindSense AI — your predictive maintenance assistant.\n\nAsk me about turbine faults, risk scores, maintenance schedules, or anything about the fleet.',
    },
  ])
  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef(null)
  const inputRef              = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 100)
  }, [isOpen])

  const send = async (text) => {
    const question = (text || input).trim()
    if (!question || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const { data } = await postChat({
        message:    question,
        turbine_id: turbineId || null,
        context:    turbineData || null,
      })
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ Error: ${err?.response?.data?.detail || err.message}`,
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  if (!isOpen) return null

  return (
    <div
      id="chatbot-panel"
      style={{
        position: 'fixed',
        bottom: 20,
        right: 20,
        width: 380,
        maxHeight: '75vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(6,11,24,0.97)',
        border: '1px solid rgba(167,139,250,0.3)',
        borderRadius: 16,
        boxShadow: '0 8px 40px rgba(0,0,0,0.6), 0 0 30px rgba(167,139,250,0.1)',
        backdropFilter: 'blur(20px)',
        zIndex: 1000,
        overflow: 'hidden',
        animation: 'fadeIn 0.2s ease',
      }}
    >
      {/* Header */}
      <div style={{
        padding: '14px 16px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'linear-gradient(135deg, rgba(124,58,237,0.15), rgba(167,139,250,0.08))',
        flexShrink: 0,
      }}>
        <div className="flex items-center gap-sm">
          <span style={{ fontSize: '1.1rem' }}>🤖</span>
          <div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>WindSense AI</div>
            {turbineId && (
              <div style={{ fontSize: '0.7rem', color: 'var(--violet)', marginTop: 1 }}>
                Context: {turbineId}
              </div>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'none', border: 'none', color: 'var(--text-muted)',
            cursor: 'pointer', fontSize: '1.1rem', lineHeight: 1, padding: 4,
          }}
        >×</button>
      </div>

      {/* Messages */}
      <div
        id="chatbot-messages"
        className="scroll-list"
        style={{ flex: 1, overflowY: 'auto', padding: '14px 12px', minHeight: 0 }}
      >
        {messages.map((m, i) => <Message key={i} msg={m} />)}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      {messages.length === 1 && (
        <div style={{ padding: '0 12px 8px', display: 'flex', flexWrap: 'wrap', gap: 4, flexShrink: 0 }}>
          {SUGGESTED.map(s => (
            <button
              key={s}
              onClick={() => send(s)}
              style={{
                background: 'rgba(167,139,250,0.08)',
                border: '1px solid rgba(167,139,250,0.2)',
                borderRadius: 100,
                color: 'var(--violet)',
                fontSize: '0.7rem',
                padding: '4px 10px',
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >{s}</button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{
        padding: '10px 12px',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        gap: 8,
        flexShrink: 0,
      }}>
        <textarea
          ref={inputRef}
          id="chatbot-input"
          className="input"
          rows={1}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask about faults, risks, maintenance…"
          disabled={loading}
          style={{ resize: 'none', fontSize: '0.85rem', padding: '8px 12px', flex: 1 }}
        />
        <button
          id="chatbot-send"
          className="btn btn--primary"
          onClick={() => send()}
          disabled={loading || !input.trim()}
          style={{ padding: '8px 14px', fontSize: '0.85rem', flexShrink: 0 }}
        >
          {loading ? '…' : '→'}
        </button>
      </div>
    </div>
  )
}
