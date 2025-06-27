# import eventlet
# eventlet.monkey_patch()



# import base64
# from redis import Redis
# from flask_socketio import SocketIO
# # from mainApp import socketio  # reusing app context
# import io
# import tempfile
# import demo1  # your mesh processing logic
# import os


# socketio = SocketIO(message_queue="redis://127.0.0.1:6379")

# # socketio = SocketIO(message_queue="redis://localhost:6379")

# def process_frame(image_bytes_b64, client_sid, job_id):
#     print(image_bytes_b64)
#     image_data = base64.b64decode(image_bytes_b64)

#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
#         img_file.write(image_data)
#         img_path = img_file.name

#     # This should run demo1.py and save mesh to `mesh_output.png`
#     mesh_path = f"/tmp/mesh_{job_id}.png"
#     try:
#         demo1.run_demo(img_path, openpose_path=None, out_path=mesh_path)
#     except Exception as e:
#         print(f"[Worker Error] {e}")
#         return

#     # Send mesh result back to client via socket
#     with open(mesh_path, "rb") as f:
#         encoded_mesh = base64.b64encode(f.read()).decode("utf-8")

#     socketio.emit("mesh_result", {
#         "job_id": job_id,
#         "image_b64": encoded_mesh
#     }, room=client_sid)

#     os.remove(mesh_path)
#     os.remove(img_path)


# if __name__ == "__main__":
#     print("RQ worker started manually.")
  # Your mesh processing logic

import sys

import base64
import os
import tempfile
from flask_socketio import SocketIO
from PIL import Image
import cv2
from io import BytesIO

print("hi")
from models import hmr


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the SPINNew directory to sys.path so we can import models/, config.py, etc.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import worker_demo as worker_demo

print("hi")



socketio = SocketIO(message_queue="redis://127.0.0.1:6379")

def process_frame(image_bytes_b64, client_sid, job_id):
    print("📥 [worker] Job received:", job_id, flush=True)

    mesh_path = None
    img_path = None

    try:
        # Decode base64 image data
        image_data = base64.b64decode(image_bytes_b64)

        # Optional sanity check with PIL
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()  # this raises an exception if the image is corrupt
            print("✅ Image verified with PIL")
        except Exception as e:
            raise ValueError(f"PIL verification failed: {e}")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
            img_file.write(image_data)
            img_path = img_file.name

        # Check if OpenCV can read the saved image
        if cv2.imread(img_path) is None:
            raise ValueError(f"OpenCV could not read the saved image at {img_path}")

        # Generate output mesh path
        mesh_path = f"/tmp/mesh_{job_id}.png"

        # Call demo pipeline
        worker_demo.run_demo(img_path, openpose_path=None, out_path=mesh_path)

        # Encode mesh output to base64
        with open(mesh_path, "rb") as f:
            encoded_mesh = base64.b64encode(f.read()).decode("utf-8")

        # Emit result to client
        socketio.emit("mesh_result", {
            "job_id": job_id,
            "image_b64": encoded_mesh
        }, room=client_sid)
        print(f"✅ [worker] Mesh result sent for job {job_id}")

    except Exception as e:
        print(f"❌ [worker] Error in job {job_id}: {e}", flush=True)
        socketio.emit("mesh_error", {
            "job_id": job_id,
            "error": str(e)
        }, room=client_sid)

    finally:
        # Clean up
        if mesh_path and os.path.exists(mesh_path):
            os.remove(mesh_path)
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

# def process_frame(image_bytes_b64, client_sid, job_id):
#     print("hi",flush=True)
#     print(f"[worker] Job received: {job_id}",flush=True)

#     try:
#         image_data = base64.b64decode(image_bytes_b64)

#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
#             img_file.write(image_data)
#             img_path = img_file.name

#         mesh_path = f"/tmp/mesh_{job_id}.png"
#         worker_demo.run_demo(img_path, openpose_path=None, out_path=mesh_path)

#         with open(mesh_path, "rb") as f:
#             encoded_mesh = base64.b64encode(f.read()).decode("utf-8")

#         socketio.emit("mesh_result", {
#             "job_id": job_id,
#             "image_b64": encoded_mesh
#         }, room=client_sid)

#         print(f"[worker] Result sent for job {job_id}")

#     except Exception as e:
#         print(f"[worker] Error: {e}")
#         socketio.emit("mesh_error", {"job_id": job_id, "error": str(e)}, room=client_sid)

#     finally:
#         if os.path.exists(mesh_path): os.remove(mesh_path)
#         if os.path.exists(img_path): os.remove(img_path)
