import { create } from 'zustand'

// API base: use Vite env variable VITE_API_BASE if set, otherwise when running
// the frontend dev server (port 5173) default to backend on localhost:8000.
const API_BASE: string =
  ((import.meta as any).env?.VITE_API_BASE as string) ||
  (location.port === '5173' ? 'http://127.0.0.1:8000' : '')

type Message = {
  id: string
  role: 'user' | 'assistant'
  text: string
  agent?: string
  provenance?: any
  checkpointId?: string
}

type State = {
  reset: () => void
  sessionId?: string
  userId?: string
  messages: Message[]
  ensureSession: () => Promise<void>
  send: (text: string) => Promise<void>
  resume: (ck: string, text: string) => Promise<void>
}

export const useChat = create<State>((set, get) => ({
  messages: [],
  reset: () => set({ messages: [], sessionId: undefined, userId: undefined }),
  async ensureSession() {
    if (get().sessionId) return
    const r = await fetch(`${API_BASE}/api/session/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    if (!r.ok) throw new Error('session create failed: ' + r.status)
    const j = await r.json()
    set({ sessionId: j.session_id, userId: j.user_id })
  },
  async send(text) {
    const { sessionId, userId } = get()
    if (!sessionId || !userId) return
    set((s) => ({
      messages: [...s.messages, { id: crypto.randomUUID(), role: 'user', text }],
    }))
    const r = await fetch(`${API_BASE}/api/chat/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, user_id: userId, text }),
    })
    if (!r.ok) throw new Error('chat send failed: ' + r.status)
    const j = await r.json()
    wsStream(j.stream_url, set)
  },
  async resume(ck, text) {
    const fd = new FormData()
    fd.set('checkpoint_id', ck)
    fd.set('text', text)
    const r = await fetch(`${API_BASE}/api/chat/resume`, { method: 'POST', body: fd })
    if (!r.ok) throw new Error('resume failed: ' + r.status)
    const j = await r.json()
    wsStream(j.stream_url, set)
  },
}))

function wsStream(url: string, set: any) {
  // compute full WebSocket URL
  let wsUrl: string
  if (url.startsWith('ws://') || url.startsWith('wss://')) {
    wsUrl = url // server already gave full ws URL
  } else {
    // API_BASE may be http(s)://host:port
    if (API_BASE) {
      const u = new URL(API_BASE)
      u.protocol = u.protocol.replace('http', 'ws')
      wsUrl = u.origin + url
    } else {
      wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + url
    }
  }

  const ws = new WebSocket(wsUrl)
  let buf = ''

  ws.onopen = () => {
    console.info('WebSocket connected:', wsUrl)
  }

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data)
    if (msg.type === 'token') {
      if (typeof msg.data === 'string') {
        buf += msg.data
      }
    }
    if (msg.type === 'meta') {
      const hasServerText =
        msg.data && typeof msg.data.text === 'string' && msg.data.text.trim().length > 0
      const finalText = hasServerText ? msg.data.text : buf || '(no output)'
      const m: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: finalText,
        agent: msg.data.agent,
        provenance: msg.data.provenance,
        checkpointId: msg.data.checkpoint_id || undefined,
      }
      set((s: any) => ({ messages: [...s.messages, m] }))
      buf = ''
    }
    if (msg.type === 'done') {
      ws.close()
    }
  }

  ws.onerror = (ev) => {
    console.error('WebSocket error', ev)
    set((s: any) => ({
      messages: [
        ...s.messages,
        { id: crypto.randomUUID(), role: 'assistant', text: '[Error streaming response]' },
      ],
    }))
  }

  ws.onclose = () => {
    console.info('WebSocket closed:', wsUrl)
  }
}
