# from time import time
import json
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
# from fastapi.params import Body
# import threading
import multiprocessing
import logging
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.websockets import WebSocketState
from .readings import collect_data
from contextlib import asynccontextmanager
# from asyncio import time
import joblib
import ipaddress

result_queue=multiprocessing.Queue()
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


# app.add_middleware(TrustedHostMiddleware, allowed_hosts=["192.168.1.*", "localhost", "127.0.0.1"])

START_IP = ipaddress.ip_address("192.168.1.100")
END_IP = ipaddress.ip_address("192.168.1.200")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global connected_client
    client_host=websocket.client.host

    try:
        host_ip=ipaddress.ip_address(client_host)
        if not START_IP<=host_ip<=END_IP:
            await websocket.close(code=1008,reason="IP not allowed")
            return
        if connected_client is not None:
            await websocket.close(code=1008, reason="Only one connection allowed.")
            return
                

    except ValueError:
        return
    

    connected_client = websocket
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'START_ANALYSIS' :
                patient_name:str = message['name']
                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': 'starting'}))
                reading_and_result=multiprocessing.Process(target=collect_data,args=("test", patient_name,result_queue), daemon=False)
                reading_and_result.start()
                reading_and_result.join()
                reading_and_result.close()
                return_result= result_queue.get()
                # if result:
                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': return_result}))
                # await websocket.close()
                # print('socket closed ')
                break

    except WebSocketDisconnect:
        print("WebSocket connection closed")
        
    finally:
        connected_client = None
        if not websocket.application_state == WebSocketState.DISCONNECTED:
            await websocket.close(code=1000, reason="Connection closed in finally block.")


# @app.get('/shutdown')
# def shutdown_jetson():




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)