 GNU nano 7.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                /home/radxa/yolodetect/drive.py
import cv2
import time
import board
import busio
import threading
from flask import Flask, Response, render_template_string
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = '/home/radxa/yolodetect/rk3588_rknn_model'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = (CAMERA_WIDTH / 2) - 30

# FLASK CONFIG
HOST_IP = '0.0.0.0'
HOST_PORT = 5000

# --- TUNING ---
ROI_VERTICAL_CUTOFF = 0.6
Kp = 0.0011
Kd = 0.0009
BASE_SPEED = 0.15
LANE_WIDTH_PIXELS = 450

# STOP SIGN LOGIC
STOP_COOLDOWN = 5.0
STOP_THRESHOLD_Y = CAMERA_HEIGHT * 0.8
BOOST_MULTIPLIER = 2
BOOST_DURATION = 2

# MOTOR PHYSICS
MIN_MOTOR_POWER = 0.07
MAX_STEER = 0.575

# --- GLOBAL VARIABLES FOR STREAMING ---
output_frame = None
lock = threading.Lock()
robot_ready = False

# --- FLASK APP ---
app = Flask(__name__)

# --- Motor Class ---
class Motor:
    def __init__(self, pca, in1, in2):
        self.pca = pca
        self.in1 = pca.channels[in1]
        self.in2 = pca.channels[in2]

    def set_speed(self, speed):
        if abs(speed) < 0.01:
            pwm = 0
        else:
            abs_s = abs(speed)
            mapped_speed = MIN_MOTOR_POWER + (abs_s * (1.0 - MIN_MOTOR_POWER))
            pwm = int(min(mapped_speed, 1.0) * 65535)

        if speed > 0:
            self.in1.duty_cycle = pwm
            self.in2.duty_cycle = 0
        elif speed < 0:
            self.in1.duty_cycle = 0
            self.in2.duty_cycle = pwm
        else:
            self.stop()

    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0

# --- ROBOT LOGIC THREAD ---
def robot_control_loop():
    global output_frame, lock

    # 1. Init Hardware
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100
        left_motors = [Motor(pca, 7, 6), Motor(pca, 4, 5)]
        right_motors = [Motor(pca, 3, 2), Motor(pca, 0, 1)]
    except Exception as e:
        print(f"Hardware Init Error: {e}")
        return

    def set_drive(fwd, steer):
        steer = max(min(steer, MAX_STEER), -MAX_STEER)
        left = fwd + steer
        right = fwd - steer
        max_val = max(abs(left), abs(right))
        if max_val > 1.0:
            left /= max_val
            right /= max_val
        for m in left_motors: m.set_speed(left)
        for m in right_motors: m.set_speed(right)

    global motors_list
    motors_list = left_motors + right_motors

    # 2. Load Model
    print("Loading YOLO Model...")
    model = YOLO(MODEL_PATH)

    prev_error = 0
    last_boost_time = 0
    boost_active = False
    boost_end_time = 0

    print("\n--- ROBOT STARTED ---")

    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            raise RuntimeError("Cannot open camera (VideoCapture(0) failed)")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Flip BEFORE YOLO inference (fix mirrored webcam)
            frame = cv2.flip(frame, 1)

            # Run inference on the flipped frame
            results = model.predict(source=frame, conf=0.3, imgsz=640, verbose=False)
            result = results[0]
            boxes = result.boxes

            # --- VISION PROCESSING ---
            best_y_x = None
            best_w_x = None
            max_y_area = 0
            max_w_area = 0
            boost_requested = False

            current_time = time.time()

            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()

                # Red Line Check - trigger speed boost
                if cls == 'redline':
                    if y > STOP_THRESHOLD_Y:
                        if (current_time - last_boost_time) > STOP_COOLDOWN:
                            boost_requested = True

                # Lane Check (Turn Later Logic)
                cutoff_pixel = CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF
                if y < cutoff_pixel:
                    continue

                area = w * h
                if cls == 'yellowline' and area > max_y_area:
                    max_y_area = area
                    best_y_x = x
                elif cls == 'whiteline' and area > max_w_area:
                    max_w_area = area
                    best_w_x = min(x, CAMERA_WIDTH -1)

            # --- VIDEO FRAME UPDATE ---
            annotated_frame = result.plot()

            # --- CONTROL LOGIC ---

            # 1. Handle boost
            #if boost_requested and not boost_active:
            #    print("!!! BOOST !!!")
            #    boost_active = True
            #    boost_end_time = current_time + BOOST_DURATION
            #    last_boost_time = current_time

            if boost_active and current_time > boost_end_time:
                boost_active = False

            # 2. Calculate Target
            if best_y_x is not None and best_w_x is not None:
                target_x = (best_y_x + best_w_x) / 2
            elif best_y_x is not None:
                target_x = best_y_x + (LANE_WIDTH_PIXELS / 2)
            elif best_w_x is not None:
                target_x = best_w_x - (LANE_WIDTH_PIXELS / 2)
            else:
                target_x = CENTER_X

            # 3. PID
            error = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error
            steering = (error * Kp) + (derivative * Kd)

            #Prevents micro-corrections on straights
            if abs(error) < 20:
                steering = 0

            # 4. Drive with boost if active
            if robot_ready:
                if boost_active:
                    current_speed = BASE_SPEED * BOOST_MULTIPLIER
                    cv2.putText(annotated_frame, "BOOSTING!", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    current_speed = BASE_SPEED

                effective_steer = steering
                set_drive(current_speed, effective_steer)

            # --- VISUAL DEBUGGING ON VIDEO ---
            debug_y = int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF) + 20
            cv2.circle(annotated_frame, (int(target_x), debug_y), 10, (0, 255, 0), -1)
            cv2.line(annotated_frame, (int(CENTER_X), 0), (int(CENTER_X), CAMERA_HEIGHT), (255, 255, 255), 1)

            with lock:
                output_frame = annotated_frame.copy()

    except Exception as e:
        print(f"Robot Loop Error: {e}")
    finally:
        stop_all()
        print("Robot Loop Ended")

def stop_all():
    for m in motors_list:
        m.stop()

# --- FLASK STREAMING FUNCTIONS ---

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Robot Vision</title>
        <style>
            body { background: #111; color: #eee; text-align: center; font-family: monospace; }
            img { border: 2px solid #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>RADXA ROBOT V11</h1>
        <p>Running: Speed Boost at Red Lines</p>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def input_loop():
    global robot_ready, BASE_SPEED
    time.sleep(6)
    print("\nReady! Press G to start, S to stop...")
    while True:
        try:
            key = input().lower()
            if key == 'g':
                BASE_SPEED = 0.15
                print("GO!")
                time.sleep(1.5)
                robot_ready = True
            elif key == 's':
                print("STOPPED! PRESS G to go again...")
                robot_ready = False
                stop_all()
                BASE_SPEED = 0
        except:
             break
# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    t = threading.Thread(target=robot_control_loop, daemon=True)
    t.start()

    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()

    print(f"Starting Web Server at http://{HOST_IP}:{HOST_PORT}")
    try:
        app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Stopping...")
