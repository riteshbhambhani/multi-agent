import { create } from 'zustand'

// API base: use Vite env variable VITE_API_BASE if set, otherwise when running
// the frontend dev server (port 5173) default to backend on localhost:8000.
const API_BASE: string = ((import.meta as any).env?.VITE_API_BASE as string) || (location.port === '5173' ? 'http://127.0.0.1:8000' : '')

type Message = { id:string; role:'user'|'assistant'; text:string; agent?:string; provenance?:any; checkpointId?:string }
type State = {
  sessionId?: string
  userId?: string
  messages: Message[]
  ensureSession:()=>Promise<void>
  send:(text:string)=>Promise<void>
  resume:(ck:string, text:string)=>Promise<void>
}

export const useChat = create<State>((set,get)=> ({
  messages: [],
  async ensureSession(){
    if(get().sessionId) return
    const payload = { }
    const r = await fetch(`${API_BASE}/api/session/create`,{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})
    if(!r.ok) throw new Error('session create failed: ' + r.status)
    const j = await r.json()
    set({sessionId:j.session_id, userId:j.user_id})
  },
  async send(text){
    const {sessionId, userId} = get()
    if(!sessionId || !userId) return
    set(s=>({messages:[...s.messages, {id:crypto.randomUUID(), role:'user', text}]}))
  const r = await fetch(`${API_BASE}/api/chat/send`,{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id:sessionId, user_id:userId, text})})
    const j = await r.json()
    wsStream(j.stream_url, set)
  },
  async resume(ck, text){
    const fd = new FormData(); fd.set('checkpoint_id', ck); fd.set('text', text)
  const r = await fetch(`${API_BASE}/api/chat/resume`,{method:'POST', body:fd})
    const j = await r.json()
    wsStream(j.stream_url, set)
  }
}))

function wsStream(url:string, set:any){
  // compute websocket base. If API_BASE points to backend, use it (ws:// or wss://),
  // otherwise fall back to current location host (works when backend is same origin).
  let wsBase: string
  if(API_BASE){
    wsBase = API_BASE.replace(/^http/, 'ws')
  } else {
    wsBase = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host
  }
  const ws = new WebSocket(wsBase + url)
  let buf = ''
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data)
    if(msg.type === 'token'){
      // msg.data may be a string (a chunk) or an object (router/state updates).
      if(typeof msg.data === 'string'){
        buf += msg.data
      } else {
        // non-string token (state update) - could be used as status, ignore for now
      }
    }
    if(msg.type === 'meta'){
      // Prefer server-provided final text when available (useful when no token stream was sent)
  // Use server-provided final text only if it's a non-empty string; otherwise fall back
  // to accumulated token buffer or a placeholder.
  const hasServerText = msg.data && (typeof msg.data.text === 'string') && msg.data.text.trim().length > 0
  const finalText = hasServerText ? msg.data.text : (buf || '(no output)')
      const m:Message = { id:crypto.randomUUID(), role:'assistant', text: finalText, agent: msg.data.agent, provenance: msg.data.provenance, checkpointId: msg.data.checkpoint_id||undefined }
      set((s:any)=>({messages:[...s.messages, m]}))
      // clear buffer so future messages start fresh
      buf = ''
    }
    if(msg.type === 'done'){ ws.close() }
  }
  ws.onerror = () => {
    set((s:any)=>({messages:[...s.messages, {id:crypto.randomUUID(), role:'assistant', text:'[Error streaming response]'}]}))
  }
}
