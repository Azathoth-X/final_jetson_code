# from time import time
import json
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
# from fastapi.params import Body
import threading
import multiprocessing

from fastapi.websockets import WebSocketState
from .readings import collect_data,result_queue
from contextlib import asynccontextmanager
# from asyncio import time
import joblib

# from files_handler import upload_to_drive

@asynccontextmanager
async def lifespan(app:FastAPI):
    global prodModel
    try:
        prodModel=joblib.load("app\ml_model\agglo_cluster_model.pkl")
    except FileNotFoundError:
        print("not model file ")
    yield
    pass










app=FastAPI(debug=True,lifespan=lifespan)
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

            if message['type'] == 'START_ANALYSIS':
                # Directly start the collection if requested
                # result = collect_data()
                reading_and_result=multiprocessing.Process(target=collect_data,args=("test","test"), daemon=False)
                reading_and_result.start()
                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': 'starting'}))
                return_result= result_queue.get()
                # if result:
                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': return_result}))
                # await websocket.close()
                # print('socket closed ')
                reading_and_result.close()
                break

    except WebSocketDisconnect:
        print("WebSocket connection closed")
        
    finally:
        connected_client = None
        if not websocket.application_state == WebSocketState.DISCONNECTED:
            await websocket.close(code=1000, reason="Connection closed in finally block.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)