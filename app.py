from gevent import monkey
monkey.patch_all()
#insecure origin treated as secure , MediaFoundation Video Capture chrome://flags/
import base64
import cv2
import numpy as np

import uuid
from flask import Flask, render_template, request, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
from gevent import monkey
from geventwebsocket.handler import WebSocketHandler
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/socket.io/*": {"origins": "http://13.233.165.48:5000"}})
app.config['SECRET_KEY'] = 'secret@123'  # Replace with your secret key
socketio = SocketIO(app,  async_mode='gevent', cors_allowed_origins="*", websocket=True)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

@app.route('/')
def index():
    session['session_id'] = str(uuid.uuid4())
    return render_template('index1.html')

# @socketio.on('stream_frame')
# def video_stream(frameData):
#     if request.environ.get('wsgi.websocket'):
#         ws = request.environ['wsgi.websocket']
#         print("===ws>", ws)
#         if 'session_id' not in session:
#             raise KeyError('Session is disconnected')
#
#         while True:
#             # Send video frames to the client
#             frame = frameData()  # Implement your logic to get video frames
#             print("===frame", frameData)
#             ws.send(frame)
#
#     return 'Invalid WebSocket request'

@socketio.on('stream_frame')
def handle_stream_frame(frameData):
    try:
        # Convert the frame data from Base64 to OpenCV image
        if 'session_id' not in session:
            raise KeyError('Session is disconnected')

        frameData = frameData.split(',')[1]
        frameBytes = base64.b64decode(frameData)
        frameArray = np.frombuffer(frameBytes, dtype=np.uint8)
        frame = cv2.imdecode(frameArray, cv2.IMREAD_COLOR)

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect smiles within each face region
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(30, 30))

            # Check if a smile is detected
            if len(smiles) > 0:
                smile_detected = True
            else:
                smile_detected = False

            # Emit the result back to the client
            socketio.emit('smile_detection_result', smile_detected)

    except KeyError as e:
        # Handle the KeyError when the session is disconnected
        # For example, you can send an error message back to the client
        socketio.emit('error', 'Session is disconnected')

    except Exception as e:
        # Handle other exceptions that might occur during smile detection
        # For example, you can log the error or send an error message back to the client
        print(f"Error during smile detection: {e}")
        socketio.emit('error', 'An error occurred during smile detection')






if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', port=5000)
    http_server = WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    http_server.serve_forever()
