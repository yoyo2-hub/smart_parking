import cv2
import pickle
import numpy as np
import os
from ultralytics import YOLO
from shapely.geometry import Polygon

# --- CONFIGURATION ---
VIDEO_PATH = "video_night_ultra_realistic.mp4"
PICKLE_FILE = "parking_slots.pkl"
OUTPUT_VIDEO = "parking_dynamic_stabilized-final2.mp4"
MODEL_PATH = "yolo11s.pt"

FRAME_SKIP = 3       
BUFFER_SIZE = 12     
OCCUPANCY_THRESHOLD = 45  

# --- INITIALIZATION ---
if not os.path.exists(PICKLE_FILE):
    raise Exception(f"File {PICKLE_FILE} not found! Run the selector first.")

with open(PICKLE_FILE, "rb") as f:
    parking_slots = pickle.load(f)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Initialize Feature Tracker for Camera Movement
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 1. Capture Reference Frame for Tracking
ret, first_frame = cap.read()
if not ret:
    exit("Could not read video.")

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

# --- VIDEO WRITER SETUP ---
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps // FRAME_SKIP, (width, height))

smoothing_buffers = {i: [] for i in range(len(parking_slots))}
full_spots_notified = set() 
frame_count = 0

print(f"Moniteur actif : Dashboard agrandi, Bulles (11), Texte (0.6).")

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    if frame_count % FRAME_SKIP != 0: continue

    # 2. TRACK CAMERA MOVEMENT
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)
    
    matches = bf.match(des_ref, des_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) > 10:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        matrix = np.eye(3)

    # 3. AI DETECTION
    results = model.predict(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.6)
    result = results[0]
    has_masks = result.masks is not None
    detections = result.masks.xy if has_masks else result.boxes.xyxy.cpu().numpy()

    annotated_frame = frame.copy()
    occupied_count = 0

    # 4. PROCESS EACH SLOT
    for idx, slot in enumerate(parking_slots):
        slot_pts_static = np.array(slot, np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(slot_pts_static, matrix)
        current_slot = transformed_pts.reshape(-1, 2).tolist()
        
        slot_poly = Polygon(current_slot)
        max_inter_area = 0
        
        for det in detections:
            if has_masks:
                if len(det) < 3: continue
                car_poly = Polygon(det)
            else:
                x1, y1, x2, y2 = det
                car_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            
            if slot_poly.intersects(car_poly):
                inter_area = slot_poly.intersection(car_poly).area
                if inter_area > max_inter_area:
                    max_inter_area = inter_area

        raw_occ = (max_inter_area / slot_poly.area) * 100
        buffer = smoothing_buffers[idx]
        buffer.append(min(100, raw_occ))
        if len(buffer) > BUFFER_SIZE: buffer.pop(0)
        smooth_occ = sum(buffer) / len(buffer)

        # 5. REFINED VISUAL DISPLAY
        is_occupied = smooth_occ > OCCUPANCY_THRESHOLD
        color = (0, 0, 255) if is_occupied else (0, 255, 0) 
        if is_occupied: occupied_count += 1
        
        center_x = int(np.mean([p[0] for p in current_slot]))
        center_y = int(np.mean([p[1] for p in current_slot]))
        
        # Bubble Scale 11
        cv2.circle(annotated_frame, (center_x, center_y), 13, (255, 255, 255), -1) 
        cv2.circle(annotated_frame, (center_x, center_y), 11, color, -1)
        
        # Text Scale 0.6
        text = f"{int(smooth_occ)}%"
        cv2.putText(annotated_frame, text, (center_x + 18, center_y + 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # --- 6. NEW LARGE HIGH-VISIBILITY DASHBOARD ---
    available_spots = len(parking_slots) - occupied_count
    
    # Define box dimensions
    box_x1, box_y1 = 20,30
    box_x2, box_y2 = 380, 180
    
    # Create semi-transparent overlay
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    # Blend with original (0.8 alpha for a "clearer" darker box)
    cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)
    
    # Add a white border around the dashboard box for clarity
    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
    
    # High-visibility text
    cv2.putText(annotated_frame, f"TOTAL SPOTS: {len(parking_slots)}", (box_x1 + 20, box_y1 + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(annotated_frame, f"OCCUPIED: {occupied_count}", (box_x1 + 20, box_y1 + 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    
    cv2.putText(annotated_frame, f"AVAILABLE: {available_spots}", (box_x1 + 20, box_y1 + 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
    
    out.write(annotated_frame)
    cv2.imshow("Dynamic Stabilized Monitor", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()