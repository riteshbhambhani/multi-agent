import { create } from 'zustand'

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
    const fd = new FormData()
    const r = await fetch('/api/session/create',{method:'POST', body:fd})
    const j = await r.json()
    set({sessionId:j.session_id, userId:j.user_id})
  },
  async send(text){
    const {sessionId, userId} = get()
    if(!sessionId || !userId) return
    set(s=>({messages:[...s.messages, {id:crypto.randomUUID(), role:'user', text}]}))
    const r = await fetch('/api/chat/send',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id:sessionId, user_id:userId, text})})
    const j = await r.json()
    wsStream(j.stream_url, set)
  },
  async resume(ck, text){
    const fd = new FormData(); fd.set('checkpoint_id', ck); fd.set('text', text)
    const r = await fetch('/api/chat/resume',{method:'POST', body:fd})
    const j = await r.json()
    wsStream(j.stream_url, set)
  }
}))

function wsStream(url:string, set:any){
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + url)
  let buf = ''
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data)
    if(msg.type === 'token'){ /* could show typing */ }
    if(msg.type === 'meta'){
      const m:Message = { id:crypto.randomUUID(), role:'assistant', text: buf || '(no output)', agent: msg.data.agent, provenance: msg.data.provenance, checkpointId: msg.data.checkpoint_id||undefined }
      set((s:any)=>({messages:[...s.messages, m]}))
    }
    if(msg.type === 'done'){ ws.close() }
  }
  ws.onerror = () => {
    set((s:any)=>({messages:[...s.messages, {id:crypto.randomUUID(), role:'assistant', text:'[Error streaming response]'}]}))
  }
}
