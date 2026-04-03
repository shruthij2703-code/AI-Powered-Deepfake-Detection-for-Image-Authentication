import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # FIX BLUE IMAGE ISSUE
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print(f"❌ FAKE ({pred:.2f})")
    else:
        print(f"✅ REAL ({1-pred:.2f})")

# Example
predict_image("test.jpg")