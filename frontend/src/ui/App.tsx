
import React, {useEffect, useMemo, useRef, useState} from 'react'
import { useChat } from './state'
import './chatgpt.css'

export default function App(){
  const s = useChat()
  useEffect(()=>{ s.ensureSession() },[])
  return (
    <div className="app">
      <Sidebar />
      <div className="main">
        <div className="topbar">
          <div className="topbarTitle">ChatGPT-like UI · Session {s.sessionId ? s.sessionId.slice(0,8) : '…'}</div>
        </div>
        <MessageList />
        <Composer />
      </div>
    </div>
  )
}

function Sidebar(){
  const s = useChat()
  return (
    <aside className="sidebar">
      <div className="brand">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M4 4h16v16H4z" stroke="#10a37f"/></svg>
        <span>Assistant</span>
      </div>
      <button className="newchat" onClick={()=>s.reset?.()}>+ New chat</button>
      <div className="history">
        <div className="historyItem">This session</div>
      </div>
      <div className="sidebarFooter">UI matches ChatGPT layout; API calls unchanged.</div>
    </aside>
  )
}

function MessageList(){
  const s = useChat()
  const ref = useRef<HTMLDivElement>(null)
  useEffect(()=>{ ref.current?.scrollTo({top: ref.current.scrollHeight}); }, [s.messages.length])
  return (
    <div className="messages" ref={ref}>
      {s.messages.map(m => (
        <div key={m.id} className={`msg role-${m.role}`}>
          <div className="msgInner">
            <div className="msgHeader">{m.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="msgText">{m.text}</div>
            {!!m.provenance && (
              <div className="provenance">
                provenance: {typeof m.provenance === 'string' ? m.provenance : JSON.stringify(m.provenance)}
              </div>
            )}
            {!!m.checkpointId && <Resume checkpointId={m.checkpointId} />}
          </div>
        </div>
      ))}
      {s.messages.length === 0 && (
        <div className="msg role-assistant">
          <div className="msgInner">
            <div className="msgHeader">Assistant</div>
            <div className="msgText">Ask me about benefits, claims, UM, or billing. I’ll answer in the style of ChatGPT.</div>
          </div>
        </div>
      )}
    </div>
  )
}

function Composer(){
  const s = useChat()
  const [txt, setTxt] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)

  // autosize
  useEffect(()=>{
    if(!taRef.current) return
    taRef.current.style.height = '24px'
    taRef.current.style.height = Math.min(180, taRef.current.scrollHeight) + 'px'
  }, [txt])

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const v = txt.trim()
    if(!v) return
    setTxt('')
    s.send(v)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      const v = txt.trim()
      if(!v) return
      setTxt('')
      s.send(v)
    }
  }

  return (
    <div className="composer">
      <form className="composerInner" onSubmit={onSubmit}>
        <div className="textareaWrap">
          <textarea
            ref={taRef}
            className="textarea"
            placeholder="Send a message…"
            value={txt}
            onChange={e=>setTxt(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button className="send" type="submit" disabled={!txt.trim()}>Send</button>
         </div>
         <div className="helperRow">
           <div className="hint">Press <span className="kbd">Enter</span> to send, <span className="kbd">Shift</span> + <span className="kbd">Enter</span> for newline</div>
         </div>
      </form>
    </div>
  )
}

function Resume({checkpointId}:{checkpointId:string}){
  const s = useChat()
  const [txt,setTxt] = useState("It's about my claim on 2024-06-14 at Riverside")
  const [busy,setBusy] = useState(false)
  return (
    <div className="resumeCard">
      <div>Need clarification? Provide more context and I'll resume.</div>
      <div className="resumeRow">
        <input className="resumeInput" value={txt} onChange={e=>setTxt(e.target.value)} />
        <button className="resumeBtn" onClick={async()=>{ setBusy(true); await s.resume(checkpointId, txt); setBusy(false); }} disabled={busy}>
          {busy ? 'Sending…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
