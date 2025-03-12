from flask import Flask, Response, render_template, send_from_directory, url_for
import cv2
import numpy as np
import threading
import datetime
import time
import os
import datetime
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import requests

app = Flask(__name__)

# Constants
FRAMES_TO_PERSIST = 10
MIN_SIZE_FOR_MOVEMENT = 1000
MOVEMENT_DETECTED_PERSISTENCE = 50

# Define the ROI (Region of Interest)
roi_x, roi_y, roi_width, roi_height = 495, 255, 210, 75

# Global variables
frame_lock = threading.Lock()
current_frame = None

# Pushover credentials
USER_KEY = "uko5ncz423kkc1ydcc2cga9xnzcn91"
API_TOKEN = "ar6h5u98tic7ngnhhe3i2p1yehq5y7"

# Video and thumbnail directories
VIDEO_DIR = 'static/videos'
THUMBNAIL_DIR = 'static/thumbnails'

# Ensure directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def send_pushover_notification(message, title=None, url=None):
    pushover_url = "https://api.pushover.net/1/messages.json"
    
    payload = {
        "token": API_TOKEN,
        "user": USER_KEY,
        "message": message
    }
    
    if title:
        payload["title"] = title
    
    if url:
        payload["url"] = url
        payload["url_title"] = "View Stream"  # This is the text that will appear on the link
    
    response = requests.post(pushover_url, data=payload)
    
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")
        print(f"Response: {response.text}")

def create_thumbnail(video_path, output_path, size=(320, 240)):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        thumbnail = cv2.resize(frame, size)
        cv2.imwrite(output_path, thumbnail)
    cap.release()

def process_frames():
    global current_frame
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (720, 480)}))
    picam2.start()

    first_frame = None
    next_frame = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0
    frame_count = 0
    last_notification_time = 0
    
    # Video recording variables
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False
    record_start_time = None

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        frame_count += 1
        if frame_count % 2 != 0:  # Process every other frame
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        roi = gray[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (255, 0, 0), 2)

        if first_frame is None:
            first_frame = roi.copy()
        
        delay_counter += 1
        if delay_counter > FRAMES_TO_PERSIST:
            delay_counter = 0
            first_frame = next_frame.copy()

        next_frame = roi.copy()

        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        largest_area = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area > MIN_SIZE_FOR_MOVEMENT:
                if area > largest_area:
                    largest_area = area
                    largest_contour = c

        if largest_contour is not None:
            cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2, offset=(roi_x, roi_y))
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

            # Start recording if not already recording
            if not recording:
                recording = True
                record_start_time = time.time()
                video_name = f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                out = cv2.VideoWriter(os.path.join(VIDEO_DIR, video_name), fourcc, 20.0, (720, 480))

            # Send Pushover notification if movement is detected and enough time has passed
            current_time = time.time()
            if current_time - last_notification_time > 30:  # (30 seconds)
                stream_url = "http://192.168.5.55:5000/"
                send_pushover_notification("Movement detected!", "Security Alert", url=stream_url)
                last_notification_time = current_time

        if movement_persistent_counter > 0:
            text = f"Movement Detected {movement_persistent_counter}"
            movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"

        cv2.putText(frame, text, (10,35), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

        frame_delta_color = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        frame_delta_color = cv2.resize(frame_delta_color, (roi_width, roi_height))
        frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = frame_delta_color

        # Write frame to video if recording
        if recording:
            out.write(frame)
            # Stop recording after 30 seconds
            if time.time() - record_start_time > 30:
                recording = False
                out.release()
                # Create thumbnail
                video_path = os.path.join(VIDEO_DIR, video_name)
                thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{os.path.splitext(video_name)[0]}.jpg")
                create_thumbnail(video_path, thumbnail_path)

        with frame_lock:
            current_frame = frame.copy()

        time.sleep(0.01)  # Small delay to reduce CPU usage

def generate_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # Adjust frame rate, approximately 30 fps
def manage_recordings():
    videos = []
    for video in os.listdir(VIDEO_DIR):
        if video.endswith('.mp4'):
            video_path = os.path.join(VIDEO_DIR, video)
            thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{os.path.splitext(video)[0]}.jpg")
            
            videos.append({
                'name': video,
                'path': video_path,
                'thumbnail': thumbnail_path,
                'date': datetime.datetime.fromtimestamp(os.path.getmtime(video_path)),
                'timestamp': os.path.getmtime(video_path)
            })
    
    # Sort videos by timestamp (most recent first)
    videos.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Keep only the 10 most recent videos
    videos_to_keep = videos[:10]
    videos_to_delete = videos[10:]
    
    # Delete older videos and their thumbnails
    for video in videos_to_delete:
        os.remove(video['path'])
        if os.path.exists(video['thumbnail']):
            os.remove(video['thumbnail'])
    
    return videos_to_keep

@app.route('/')
def index():
    videos = manage_recordings()
    
    video_data = []
    for video in videos:
        if not os.path.exists(video['thumbnail']):
            create_thumbnail(video['path'], video['thumbnail'])
        
        video_data.append({
            'name': video['name'],
            'thumbnail': url_for('static', filename=f'thumbnails/{os.path.basename(video["thumbnail"])}'),
            'date': video['date'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return render_template('index.html', videos=video_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/<path:filename>')
def download_video(filename):
    return send_from_directory(VIDEO_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

@app.route('/delete_all', methods=['POST'])
def delete_all():
    # Implement the logic to delete all recordings here
    delete_all_recordings()
    return redirect(url_for('index'))

def delete_all_recordings():
    for file in os.listdir(VIDEO_DIR):
        if file.endswith('.mp4'):
            os.remove(os.path.join(VIDEO_DIR, file))
    for file in os.listdir(THUMBNAIL_DIR):
        if file.endswith('.jpg'):
            os.remove(os.path.join(THUMBNAIL_DIR, file))