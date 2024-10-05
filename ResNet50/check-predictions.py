import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = tf.keras.models.load_model('wildfire_detection_model_resnet50.keras')

# ฟังก์ชันทำนายผลจากภาพ
def predict_images(model, image_dir, num_images):
    predictions = []
    image_files = os.listdir(image_dir)[:num_images]  # เลือกเฉพาะจำนวนภาพที่ต้องการ
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # ทำการ normalize

        # ทำนายผลลัพธ์
        prediction = model.predict(img_array)
        predictions.append(prediction[0][0])

    return predictions

# กำหนดเส้นทางไปยังโฟลเดอร์ที่มีภาพแต่ละคลาส (fire และ nofire)
fire_image_dir = 'E:/GitHub/CNN-Wildfire-Detection/forest_fire/Testing/fire'  # โฟลเดอร์ภาพไฟป่า
nofire_image_dir = 'E:/GitHub/CNN-Wildfire-Detection/forest_fire/Testing/nofire'  # โฟลเดอร์ภาพไม่มีไฟป่า

# ทำนายผล 10 ภาพจากแต่ละคลาส
fire_predictions = predict_images(model, fire_image_dir, num_images=22)
nofire_predictions = predict_images(model, nofire_image_dir, num_images=46)

# หาค่าเฉลี่ย predictions ของแต่ละคลาส
fire_mean_prediction = np.mean(fire_predictions)
nofire_mean_prediction = np.mean(nofire_predictions)

print(f"ค่าเฉลี่ย predictions สำหรับคลาส 'fire': {fire_mean_prediction:.4f}")
print(f"ค่าเฉลี่ย predictions สำหรับคลาส 'no fire': {nofire_mean_prediction:.4f}")
