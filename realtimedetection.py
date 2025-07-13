import cv2
from keras.models import model_from_json
import numpy as np
import os

# Load model architecture
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Check working directory and files
print("📁 Current directory:", os.getcwd())
print("📄 Files here:", os.listdir())

# Load weights
model.load_weights("facialemotionmodel.weights.h5")
print("✅ Model and weights loaded successfully.")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    success, frame = webcam.read()
    if not success:
        print("⚠️ Failed to capture image from webcam.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)
        prediction = model.predict(img)
        label = labels[prediction.argmax()]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-time Facial Emotion Detection", frame)

    # Press Esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
