# ============================================================
# MOODIFY AI
# Facial Emotion Recognition + Music Recommendation
# Version 2.0
# ============================================================

import cv2
import time
import numpy as np

from tensorflow.keras.models import load_model

from playlist import get_playlist

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 50)
print("Loading Moodify AI...")
print("=" * 50)

model = load_model("best_model.keras")

print("AI Model Loaded Successfully")
print("=" * 50)

# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    print("ERROR : Haar Cascade tidak ditemukan!")

    exit()

# ============================================================
# LABEL EMOSI
# ============================================================

emotion_labels = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "sad",

    "surprise"

]

# ============================================================
# EMOJI
# ============================================================

emoji = {

    "happy":"😊",

    "sad":"😢",

    "angry":"😠",

    "neutral":"😐",

    "fear":"😨",

    "surprise":"😲",

    "disgust":"🤢"

}

# ============================================================
# WARNA
# ============================================================

colors = {

    "happy":(0,255,0),

    "sad":(255,120,0),

    "angry":(0,0,255),

    "neutral":(180,180,180),

    "fear":(255,0,255),

    "surprise":(0,255,255),

    "disgust":(0,165,255)

}

# ============================================================
# CAMERA
# ============================================================

camera_index = 0

cap = cv2.VideoCapture(camera_index)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

if not cap.isOpened():

    print("Camera gagal dibuka")

    exit()

print("Camera Connected")

# ============================================================
# FPS
# ============================================================

prev_time = time.time()

current_emotion = "neutral"

confidence = 0

playlist = get_playlist(current_emotion)

# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_face(face):

    face = cv2.resize(face,(224,224))

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = face.astype(np.float32)

    face = face / 255.0

    face = np.expand_dims(
        face,
        axis=0
    )

    return face

# ============================================================
# DRAW INFO PANEL
# ============================================================

def draw_panel(frame,
               emotion,
               confidence,
               playlist,
               fps):

    h,w,_ = frame.shape

    panel_x = 820

    cv2.rectangle(

        frame,

        (panel_x,0),

        (1280,720),

        (35,35,35),

        -1

    )

    cv2.putText(

        frame,

        "MOODIFY AI",

        (panel_x+20,40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255,255,255),

        2

    )

    cv2.putText(

        frame,

        "Emotion",

        (panel_x+20,90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (200,200,200),

        2

    )

    cv2.putText(

        frame,

        emotion.upper(),

        (panel_x+20,130),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        colors[emotion],

        3

    )

    cv2.putText(

        frame,

        f"Confidence : {confidence:.2f}%",

        (panel_x+20,170),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255,255,255),

        2

    )

    cv2.putText(

        frame,

        f"FPS : {int(fps)}",

        (panel_x+20,210),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255,255,255),

        2

    )

    cv2.putText(

        frame,

        "Recommended Songs",

        (panel_x+20,270),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0,255,255),

        2

    )

    y = 320

    for song in playlist:

        text = f"{song['artist']} - {song['title']}"

        cv2.putText(

            frame,

            text,

            (panel_x+20,y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (255,255,255),

            1

        )

        y += 35

    cv2.putText(

        frame,

        "Q : Quit",

        (panel_x+20,650),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (180,180,180),

        2

    )

    cv2.putText(

        frame,

        "S : Screenshot",

        (panel_x+20,685),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (180,180,180),

        2

    )

    return frame

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # DETEKSI WAJAH
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.2,

        minNeighbors=6,

        minSize=(100,100)

    )

    # --------------------------------------------------------
    # PREDIKSI EMOSI
    # --------------------------------------------------------

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        try:

            input_face = preprocess_face(face)

            prediction = model.predict(
                input_face,
                verbose=0
            )[0]

            emotion_index = np.argmax(prediction)

            current_emotion = emotion_labels[emotion_index]

            confidence = float(
                prediction[emotion_index] * 100
            )

            playlist = get_playlist(
                current_emotion
            )

        except Exception as e:

            print(e)

            continue

        # ----------------------------------------------------
        # DRAW BOX
        # ----------------------------------------------------

        color = colors[current_emotion]

        cv2.rectangle(

            frame,

            (x,y),

            (x+w,y+h),

            color,

            3

        )

        cv2.rectangle(

            frame,

            (x,y-35),

            (x+w,y),

            color,

            -1

        )

        label = f"{current_emotion.upper()}"

        cv2.putText(

            frame,

            label,

            (x+10,y-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255,255,255),

            2

        )

        cv2.putText(

            frame,

            f"{confidence:.1f}%",

            (x,y+h+30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # --------------------------------------------------------
    # PANEL KANAN
    # --------------------------------------------------------

    frame = draw_panel(

        frame,

        current_emotion,

        confidence,

        playlist,

        fps

    )

    # --------------------------------------------------------
    # JUDUL
    # --------------------------------------------------------

    cv2.putText(

        frame,

        "Facial Emotion Recognition",

        (20,40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255,255,255),

        2

    )

    cv2.imshow(

        "Moodify AI",

        frame

    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

    # ============================================================
# KEYBOARD CONTROL
# ============================================================

    # Tombol S = Screenshot
    if key == ord("s"):

        filename = f"screenshot_{int(time.time())}.png"

        cv2.imwrite(filename, frame)

        print(f"Screenshot disimpan : {filename}")

# ============================================================
# RELEASE
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("="*50)
print("Moodify AI Closed")
print("="*50)