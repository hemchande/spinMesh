import eventlet
eventlet.monkey_patch()
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from redis import Redis
from rq import Queue
import uuid
import os



import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from queue2.worker_module import process_frame


app = Flask(__name__)
CORS(app)
redis_url = "rediss://red-d1es52fdiees73crkc6g:3ie0gE6r6y3W9lpBOPrHMs329Z4lGkcN@oregon-keyvalue.render.com:6379"
redis_conn_NEW = Redis.from_url(redis_url)
socketio = SocketIO(app, cors_allowed_origins="*",message_queue="redis://127.0.0.1:6379")
redis_conn = Redis(host="127.0.0.1", port=6379)
job_queue = Queue(connection=redis_conn_NEW)

# Store client sid to send result later
connected_clients = {}

@socketio.on("connect")
def handle_connect():
    print(f"🔌 Client connected: {request.sid}")
    connected_clients[request.sid] = request.sid

@socketio.on("disconnect")
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")
    connected_clients.pop(request.sid, None)

import base64
from PIL import Image
import io
import os
import uuid

def save_base64_image(image_bytes, out_dir="/tmp"):
    try:
        # Handle base64 string like "data:image/jpeg;base64,/9j/..."
        header, base64_str = image_bytes.split(",", 1)
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        os.makedirs(out_dir, exist_ok=True)
        filename = f"frame_{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(out_dir, filename)
        img.save(img_path)
        print(f"✅ Image saved to: {img_path}")
        return img_path
    except Exception as e:
        print(f"❌ Failed to save image: {e}")
        return None


@socketio.on("send_frame")
def handle_frame(data):
    print(data)
    image_bytes = data.get("image_data")
    # if not image_bytes:
    #     print("true")
    #     emit("error", {"message": "Missing image data"})
    #     return
    if not image_bytes:
        emit("error", {"message": "Missing image data"})
        return

    # Remove base64 prefix if present
    if image_bytes.startswith("data:image"):
        image_bytes = image_bytes.split(",", 1)[-1]

    job_id = str(uuid.uuid4())
    job = job_queue.enqueue(process_frame, image_bytes, request.sid, job_id)
    print(job)
    emit("job_submitted", {"job_id": job_id})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8002,debug=True)

