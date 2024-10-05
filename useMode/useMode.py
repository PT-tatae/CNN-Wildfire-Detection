from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model('wildfire_detection_model.keras')

import numpy as np
from tensorflow.keras.preprocessing import image

# โหลดและประมวลผลภาพใหม่
img_path = 'E:/GitHub/CNN-Wildfire-Detection/forest_fire/Testing/nofire/abc363.jpg'  # แทนที่ด้วยเส้นทางของภาพที่คุณต้องการทดสอบ
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติให้กับข้อมูลภาพ

# ทำการทำนาย
predictions = model.predict(img_array)

# แสดงผลการทำนาย
if predictions[0] > 0.5:
    print("Predicted: Fire")
else:
    print("Predicted: No Fire")


# แสดงผลการทำนายและความน่าจะเป็น
predicted_class = "Fire" if predictions[0] > 0.5 else "No Fire"
confidence = predictions[0] if predictions[0] > 0.5 else 1 - predictions[0]
# ใช้ .item() เพื่อแปลงค่าจาก numpy.ndarray เป็น scalar
print(f"Predicted: {predicted_class} with confidence: {confidence.item():.2f}")