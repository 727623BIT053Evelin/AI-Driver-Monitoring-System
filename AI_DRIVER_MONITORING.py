import cv2
import numpy as np
from ultralytics import YOLO
import time
from pygame import mixer
import threading
import os
from datetime import datetime
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
import pyttsx3
from telegram import Bot
from telegram.request import HTTPXRequest
import asyncio
from collections import deque


# ===================== INITIALIZATIONS =====================
print("\n=== Enhanced Traffic Violation Detection System ===")
print("=== Developed by Techie Sparks ===")
os.makedirs('alerts', exist_ok=True)

# Audio alerts
mixer.init()
try:
    if not os.path.exists('audio/alert.wav'):
        open('audio/alert.wav', 'wb').close()
    drowsy_sound = mixer.Sound('audio/alert.wav')
    print("\n[STATUS] Audio alerts initialized")
except Exception as e:
    print(f"\n[ERROR] Audio alert initialization: {e}")
    drowsy_sound = None

# Voice alerts
voice_engine = None
try:
    voice_engine = pyttsx3.init()
    voice_engine.setProperty('rate', 150)
    voice_engine.setProperty('volume', 0.9)
    print("[STATUS] Voice alerts initialized")
except Exception as e:
    print(f"[ERROR] Voice alert initialization: {e}")

# Telegram setup with larger connection pool
TELEGRAM_BOT_TOKEN = '7514704187:AAGaz9pqf1KrXiMQgs5p1-pNv4nkcry-dao'
TELEGRAM_CHAT_ID = '7094249861'
telegram_bot = None
try:
    telegram_bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        request=HTTPXRequest(connection_pool_size=10, read_timeout=20, write_timeout=20)
    )
    print("[STATUS] Telegram bot initialized with enhanced connection pool")
except Exception as e:
    print(f"[ERROR] Telegram initialization: {str(e)[:100]}")

# Violation tracking
violation_count = {
    'PHONE': {'count': 0, 'fine': 1000},
    'SMOKING': {'count': 0, 'fine': 500},
    'LIQUOR': {'count': 0, 'fine': 2000},
    'DROWSY': {'count': 0, 'fine': 0},
    'OVERLOADED': {'count': 0, 'fine': 1000},
    'FOOD': {'count': 0, 'fine': 400}
}

# Message queue for Telegram alerts
telegram_queue = deque()
queue_lock = threading.Lock()
SEND_INTERVAL = 2  # Minimum 2 seconds between Telegram sends

# Person counting variables
MAX_PERSONS = 1  # Only 1 person (driver) allowed
person_count = 0

# ===================== DETECTION SETUP =====================
try:
    model = YOLO('yolov8m.pt')
    print("[STATUS] YOLO model loaded successfully")
except Exception as e:
    print(f"[ERROR] YOLO model loading: {e}")
    exit()

CLASS_MAPPING = {
    'cell phone': 'PHONE',
    'mobile phone': 'PHONE',
    'cigarette': 'SMOKING',
    'vape': 'SMOKING',
    'cigar': 'SMOKING',
    'smoking': 'SMOKING',
    'cigarette pack': 'SMOKING',
    'lighter': 'SMOKING',
    'bottle': 'LIQUOR',
    'wine bottle': 'LIQUOR',
    'beer bottle': 'LIQUOR',
    'wine glass': 'LIQUOR',
    'person': 'PERSON',
    'toothbrush': 'SMOKING',
    'pizza': 'FOOD',
    'donut': 'FOOD',
    'cake': 'FOOD',
    'sandwich': 'FOOD',
    'banana': 'FOOD',
    'apple': 'FOOD'
}

# Drowsiness detection
EYE_AR_THRESHOLD = 0.25
FPS = 30
EYE_CLOSED_SECONDS = 1
EYE_AR_CONSEC_FRAMES = FPS * EYE_CLOSED_SECONDS
COUNTER = 0

# Face landmarks
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("[STATUS] Face detection models loaded")
except Exception as e:
    print(f"[ERROR] Face model loading: {e}")
    exit()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# ===================== CORE FUNCTIONS =====================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def play_alert_then_speak(alert_message, is_drowsy=False):
    def alert_sequence():
        try:
            if is_drowsy and drowsy_sound:
                drowsy_sound.play()
                while mixer.get_busy():
                    time.sleep(0.1)
            if voice_engine:
                voice_engine.say(alert_message)
                voice_engine.runAndWait()
        except Exception as e:
            print(f"[WARNING] Alert playback: {e}")
    threading.Thread(target=alert_sequence, daemon=True).start()

async def async_send_telegram_alert(violation_type, frame, is_drowsy=False):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alerts/{violation_type}_{timestamp}.jpg"
            
            alert_frame = frame.copy()
            cv2.putText(alert_frame, f"VIOLATION: {violation_type}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if not is_drowsy:
                fine_amount = violation_count[violation_type]['fine']
                cv2.putText(alert_frame, f"FINE: ₹{fine_amount}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(alert_frame, datetime.now().strftime("%H:%M:%S"), (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(filename, alert_frame)
            
            if is_drowsy:
                message = f"⚠️ DROWSINESS WARNING (1+ second) ⚠️\nTime: {datetime.now().strftime('%H:%M:%S')}"
            else:
                fine_amount = violation_count[violation_type]['fine']
                message = (f"🚨 VIOLATION DETECTED\n"
                          f"Type: {violation_type}\n"
                          f"Fine: ₹{fine_amount}\n"
                          f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            with open(filename, 'rb') as photo:
                await telegram_bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=photo,
                    caption=message
                )
            os.remove(filename)
            return True
            
        except Exception as e:
            print(f"[WARNING] Telegram attempt {attempt+1} for {violation_type}: {str(e)[:100]}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    
    print(f"[ERROR] Failed to send {violation_type} alert after {max_retries} attempts")
    return False

def telegram_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        if telegram_queue:
            with queue_lock:
                violation_type, frame, is_drowsy = telegram_queue.popleft()
            
            try:
                success = loop.run_until_complete(
                    async_send_telegram_alert(violation_type, frame, is_drowsy)
                )
                
                if not success:
                    # Requeue if failed
                    with queue_lock:
                        telegram_queue.appendleft((violation_type, frame, is_drowsy))
            except Exception as e:
                print(f"[WORKER ERROR] {e}")
                with queue_lock:
                    telegram_queue.appendleft((violation_type, frame, is_drowsy))
        
        time.sleep(SEND_INTERVAL)

def send_telegram_alert(violation_type, frame, is_drowsy=False):
    with queue_lock:
        telegram_queue.append((violation_type, frame.copy(), is_drowsy))

def draw_bounding_box(frame, box, label, confidence, violation_type):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if violation_type != 'PERSON' else (0, 255, 0)
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Special cases for display labels
    if label.lower() == 'toothbrush':
        display_label = 'cigarette'
    elif violation_type == 'FOOD':
        display_label = 'EATING'
    elif violation_type == 'LIQUOR':
        display_label = 'LIQUOR'
    else:
        display_label = label
    
    label_text = f"{display_label} ({confidence:.2f})"
    
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

# ===================== MAIN LOOP =====================
def main():
    global person_count
    
    # Start Telegram worker thread
    worker_thread = threading.Thread(target=telegram_worker, daemon=True)
    worker_thread.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Driver Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Driver Monitoring", 1280, 720)

    last_alert_time = {v: 0 for v in violation_count}
    alert_cooldown = 60
    drowsy_cooldown = 300

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Camera feed lost")
                break
            
            current_time = time.time()
            violations = set()
            person_count = 0
            
            try:
                results = model(frame, verbose=False, conf=0.7)
                
                # Process detections
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        label = model.names[cls]
                        confidence = float(box.conf.item())
                        
                        # Person counting
                        if 'person' in label.lower():
                            person_count += 1
                            frame = draw_bounding_box(frame, box.xyxy[0], label, confidence, 'PERSON')
                            continue
                        
                        # Special detection with lower threshold
                        min_confidence = 0.6 if any(x in label.lower() for x in 
                                                   ['cigarette', 'vape', 'pen', 'smoking', 
                                                    'cigar', 'lighter', 'toothbrush',
                                                    'pizza', 'donut', 'cake', 'hot dog', 
                                                    'sandwich', 'banana', 'apple',
                                                    'bottle', 'wine bottle', 'beer bottle', 'wine glass']) else 0.7
                        
                        if confidence > min_confidence:
                            violation = None
                            for obj, violation_type in CLASS_MAPPING.items():
                                if obj in label.lower():
                                    violation = violation_type
                                    break
                            
                            if violation:
                                violations.add(violation)
                                frame = draw_bounding_box(frame, box.xyxy[0], label, confidence, violation)
                
                # Check for overloading (more than 1 person)
                if person_count > MAX_PERSONS:
                    violations.add('OVERLOADED')
                    cv2.putText(frame, f"OVERLOADED: {person_count} persons", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if current_time - last_alert_time['OVERLOADED'] > alert_cooldown:
                        play_alert_then_speak(f"Warning! Vehicle overloaded with {person_count} persons")
                
                # Send alerts for all violations
                for violation in violations:
                    if current_time - last_alert_time[violation] > alert_cooldown:
                        violation_count[violation]['count'] += 1
                        send_telegram_alert(violation, frame.copy())
                        
                        # Custom voice alerts for different violations
                        if violation == 'FOOD':
                            play_alert_then_speak("Warning! Eating while driving detected")
                        elif violation == 'PHONE':
                            play_alert_then_speak("Warning! Mobile phone usage detected")
                        elif violation == 'SMOKING':
                            play_alert_then_speak("Warning! Smoking is detected")  # Modified this line only
                        elif violation == 'LIQUOR':
                            play_alert_then_speak("Warning! Liquor detected")
                        elif violation == 'OVERLOADED':
                            pass  # Already handled above
                        
                        last_alert_time[violation] = current_time
                
                # Drowsiness detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 0)
                
                for face in faces:
                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    
                    left_eye = shape[lStart:lEnd]
                    right_eye = shape[rStart:rEnd]
                    
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                    cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1]-200, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if ear < EYE_AR_THRESHOLD:
                        COUNTER += 1
                        time_remaining = max(0, EYE_CLOSED_SECONDS - (COUNTER / FPS))
                        cv2.putText(frame, f"Drowsy Alert in: {time_remaining:.1f}s", 
                                   (frame.shape[1]-250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            if current_time - last_alert_time['DROWSY'] > drowsy_cooldown:
                                last_alert_time['DROWSY'] = current_time
                                violation_count['DROWSY']['count'] += 1
                                send_telegram_alert('DROWSY', frame.copy(), True)
                                play_alert_then_speak("Warning! Driver appears drowsy", is_drowsy=True)
                                cv2.putText(frame, "DROWSINESS WARNING!", (frame.shape[1]//2-200, 100), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    else:
                        COUNTER = 0
                        cv2.putText(frame, "Status: Eyes Open", (frame.shape[1]-200, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display information
                y_offset = 120
                cv2.putText(frame, f"Persons in vehicle: {person_count}/{MAX_PERSONS}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if person_count <= MAX_PERSONS else (0, 0, 255), 2)
                y_offset += 30
                
                if any(data['count'] > 0 for data in violation_count.values()):
                    cv2.putText(frame, "Traffic Violation Detected:", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
                    
                    for violation, data in violation_count.items():
                        if data['count'] > 0:
                            cv2.putText(frame, f"- {violation}: {data['count']}", (15, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_offset += 30
                
                cv2.imshow("Driver Monitoring", frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break

            except Exception as e:
                print(f"[ERROR] Processing frame: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[STATUS] Shutting down gracefully...")
    finally:
        print("\n=== VIOLATION SUMMARY ===")
        total_fine = 0
        for violation, data in violation_count.items():
            if data['count'] > 0:
                if violation == 'DROWSY':
                    print(f"{violation}: {data['count']} warnings")
                else:
                    fine = data['count'] * data['fine']
                    total_fine += fine
                    print(f"{violation}: {data['count']} times (Fine: ₹{fine})")
        
        print(f"\nMaximum persons detected at once: {person_count}")
        if total_fine > 0:
            print(f"\nTOTAL FINE: ₹{total_fine}")
        
        cap.release()
        cv2.destroyAllWindows()
        if voice_engine:
            voice_engine.stop()
        print("\n=== System shutdown complete ===")

if __name__ == "__main__":
    main()