import requests
import websocket
import json
import time

BACKEND = "http://127.0.0.1:8000"

print('creating session...')
r = requests.post(BACKEND + "/api/session/create", json={})
print('session create status', r.status_code, r.text)
js = r.json()
session_id = js['session_id']; user_id = js['user_id']
print('session', session_id, 'user', user_id)

print('posting chat/send')
r2 = requests.post(BACKEND + "/api/chat/send", json={"session_id": session_id, "user_id": user_id, "text": "Test streaming from integration test"})
print('chat send status', r2.status_code, r2.text)
js2 = r2.json()
stream_url = js2['stream_url']
print('stream_url', stream_url)

ws_url = 'ws://127.0.0.1:8000' + stream_url
print('connecting to', ws_url)

messages = []

def on_message(ws, message):
    print('RECV:', message)
    try:
        m = json.loads(message)
    except Exception:
        m = {'raw': message}
    messages.append(m)
    if m.get('type') == 'done':
        print('received done, closing')
        ws.close()

def on_error(ws, error):
    print('WS ERROR:', error)

def on_close(ws, code, reason):
    print('WS CLOSED', code, reason)

def on_open(ws):
    print('WS OPEN')

ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
ws.run_forever()

print('final messages:', messages)
