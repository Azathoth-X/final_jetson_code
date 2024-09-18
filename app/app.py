from time import time
import json
from fastapi import FastAPI,Response, status, HTTPException, Depends, UploadFile,WebSocket,WebSocketDisconnect
from fastapi.params import Body
import threading
from .readings import collect_data
# from files_handler import upload_to_drive

app=FastAPI(debug=True)
connected_client = None

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global connected_client

    if connected_client is not None:
        await websocket.close(code=1008, reason="Only one connection allowed.")
        return

    connected_client = websocket
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # If message contains the future time for data collection
            # if message['type'] == 'SYNC_START_TIME':
            #     future_epoch_time = message['start_time']
            #     print(f"Received future start time: {future_epoch_time}")
                
            #     # Set the global start time
            #     global_start_time = future_epoch_time

            #     # Send acknowledgment back to the server
            #     await websocket.send_text(json.dumps({'type': 'TIME_SYNC_ACK', 'status': 'completed'}))

            #     # Start a background thread to wait until the start time and start collection
            #     threading.Thread(target=collect_data(), daemon=True).start()

            if message['type'] == 'START_ANALYSIS':
                # Directly start the collection if requested
                # result = collect_data()
                result=threading.Thread(target=collect_data(), daemon=True).start()
                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': 'starting'}))
                if result:
                    await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': result}))



    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        connected_client = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)