import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# โหลดโมเดล
model = load_model('wildfire_detection_model_finetuned.keras')

# ฟังก์ชันในการทำนายภาพและแสดงผล
def predict_images(img_folder):
    img_files = os.listdir(img_folder)

    # แสดงผลลัพธ์
    plt.figure(figsize=(15, 10))

    for i, img_name in enumerate(img_files):
        img_path = os.path.join(img_folder, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        # ตรวจสอบจำนวนคลาสที่ได้จากโมเดล
        if predictions.shape[1] == 1:
            fire_confidence = predictions[0][0]  # ค่า confidence สำหรับ 'fire'
            nofire_confidence = 1 - fire_confidence  # คำนวณค่า confidence สำหรับ 'no fire'
        else:
            fire_confidence = predictions[0][0]  # ค่า confidence สำหรับ 'fire'
            nofire_confidence = predictions[0][1]  # ค่า confidence สำหรับ 'no fire'

        # กำหนด label ตามค่าความมั่นใจ
        label = "Fire" if fire_confidence <= 0.75 else "No Fire"

        # แสดงผลภาพและผลลัพธ์
        plt.subplot(5, 4, i + 1)
        plt.imshow(img)
        plt.title(f"{label} (Fire: {fire_confidence:.2f}, No Fire: {nofire_confidence:.2f})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# เรียกใช้ฟังก์ชันโดยระบุเส้นทางของโฟลเดอร์ที่มีภาพทดสอบ
predict_images('E:/GitHub/CNN-Wildfire-Detection/imgs-test')
