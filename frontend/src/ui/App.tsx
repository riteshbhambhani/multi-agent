import React, {useEffect, useState} from 'react'
import { useChat } from './state'

export default function App(){
  const s = useChat()
  useEffect(()=>{ s.ensureSession() },[])
  return (
    <div style={{maxWidth:800, margin:"40px auto", fontFamily:"Inter, system-ui"}}>
      <h2>Healthcare Benefits & Claims Assistant</h2>
      <MessageList />
      <Composer />
    </div>
  )
}

function MessageList(){
  const {messages} = useChat()
  return (
    <div style={{border:"1px solid #ddd", padding:16, borderRadius:12, minHeight:300}}>
      {messages.map(m=> (
        <div key={m.id} style={{margin:"8px 0"}}>
          <div><b>{m.role === 'user' ? 'You' : (m.agent || 'Assistant')}</b></div>
          <div style={{whiteSpace:"pre-wrap"}}>{m.text}</div>
          {m.provenance && <details><summary>Provenance</summary><pre>{JSON.stringify(m.provenance,null,2)}</pre></details>}
          {m.checkpointId && <div style={{padding:8, background:"#fffbe6", border:"1px solid #f0e6a6"}}>
            Clarification needed. <Resume checkpointId={m.checkpointId} />
          </div>}
        </div>
      ))}
    </div>
  )
}

function Composer(){
  const [text,setText] = useState("")
  const s = useChat()
  return (
    <form onSubmit={e=>{e.preventDefault(); s.send(text); setText("")}} style={{display:"flex", gap:8, marginTop:12}}>
      <input value={text} onChange={e=>setText(e.target.value)} placeholder="Ask about benefits or claims..." style={{flex:1, padding:12,borderRadius:8,border:"1px solid #ccc"}} />
      <button type="submit" style={{padding:"12px 16px"}}>Send</button>
    </form>
  )
}

function Resume({checkpointId}:{checkpointId:string}){
  const s = useChat()
  const [txt,setTxt] = useState("It's about my claim on 2024-06-14 at Riverside")
  return (
    <form onSubmit={e=>{e.preventDefault(); s.resume(checkpointId, txt)}}>
      <input value={txt} onChange={e=>setTxt(e.target.value)} style={{width:"100%", padding:8, marginTop:8}}/>
      <button type="submit" style={{marginTop:8}}>Send clarification</button>
    </form>
  )
}
