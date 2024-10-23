import json
from fastapi import FastAPI,WebSocket,WebSocketDisconnect,Request,Response,status
import multiprocessing
import logging
from fastapi.websockets import WebSocketState
from .readings import collect_data
from contextlib import asynccontextmanager
from .schema import ResultInfoModel
import ipaddress
import os
import signal
import subprocess

result_queue=multiprocessing.Queue()

@asynccontextmanager
async def lifespan(app:FastAPI):
    
    yield
    # subprocess.run(["shutdown", "-h"])
    return




app=FastAPI(debug=True,lifespan=lifespan)
connected_client = None
shutdownAvailable: bool=True


START_IP = ipaddress.ip_address("192.168.1.100")
END_IP = ipaddress.ip_address("192.168.1.200")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global connected_client, shutdownAvailable

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
    shutdownAvailable=False

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

                return_result:ResultInfoModel= result_queue.get()

                await websocket.send_text(return_result.model_dump_json())

                break

    except WebSocketDisconnect:
        print("WebSocket connection closed")
        
    finally:
        shutdownAvailable=True
        connected_client = None
        if not websocket.application_state == WebSocketState.DISCONNECTED:
            await websocket.close(code=1000, reason="Connection closed in finally block.")


@app.get('/shutdown')
def shutdown_jetson(request:Request):
    client_ip = ipaddress.IPv4Address(str(request.client.host))
    if not START_IP<=client_ip<=END_IP:
        return Response(content=f'Host not allowed{client_ip}',status_code=status.HTTP_401_UNAUTHORIZED)
    if shutdownAvailable:
        os.kill(os.getpid(),signal.SIGINT)
        return Response("Shutting Down",status.HTTP_200_OK)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)