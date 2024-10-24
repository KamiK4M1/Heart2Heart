from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import base64

app = FastAPI()

# เปิดใช้งาน CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initializing MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def is_thumb_up(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    return thumb_tip.y < thumb_mcp.y

def is_handshake(hands_landmarks):
    if len(hands_landmarks) == 2:
        hand1 = hands_landmarks[0].landmark
        hand2 = hands_landmarks[1].landmark
        hand1_index_tip = hand1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        hand2_index_tip = hand2[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = np.sqrt(
            (hand1_index_tip.x - hand2_index_tip.x) ** 2 +
            (hand1_index_tip.y - hand2_index_tip.y) ** 2
        )
        return distance < 0.1
    return False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        image_data = base64.b64decode(data)
        np_image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            if is_handshake(results.multi_hand_landmarks):
                await websocket.send_text("handshake detected")
            elif any(is_thumb_up(hand.landmark) for hand in results.multi_hand_landmarks):
                await websocket.send_text("thumbs up detected")
            else:
                await websocket.send_text("no specific pose detected")
        else:
            await websocket.send_text("no hand detected")
