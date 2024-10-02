# data_preprocessing.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir):
    # สร้าง ImageDataGenerator สำหรับการเพิ่มข้อมูล (augmentation) และการโหลดภาพ
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # ปรับขนาดของค่าสีในภาพจาก [0, 255] เป็น [0, 1]
        validation_split=0.2,        # แบ่ง 20% ของข้อมูลสำหรับการตรวจสอบ (Validation)
        rotation_range=20,           # หมุนภาพแบบสุ่มได้สูงสุด 20 องศา
        width_shift_range=0.2,       # เลื่อนภาพในแนวนอนได้สูงสุด 20% ของความกว้างภาพ
        height_shift_range=0.2,      # เลื่อนภาพในแนวตั้งได้สูงสุด 20% ของความสูงภาพ
        shear_range=0.2,             # ทำการบิดภาพ (shear) ได้สูงสุด 20%
        zoom_range=0.2,              # ซูมภาพแบบสุ่มได้สูงสุด 20%
        horizontal_flip=True         # พลิกภาพในแนวนอนแบบสุ่ม
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # สร้างตัวสร้างข้อมูลสำหรับการฝึกและการตรวจสอบ
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, validation_generator, test_generator
