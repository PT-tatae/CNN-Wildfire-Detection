# main.py
from data_preprocessing import load_data
from model_training import create_model, train_model

# กำหนดเส้นทางของ dataset
train_dir = 'forest_fire/Training and Validation'
test_dir = 'forest_fire/Testing'

# โหลดข้อมูล
train_generator, validation_generator, test_generator = load_data(train_dir, test_dir)

# สร้างโมเดล
model = create_model()

# ฝึกโมเดล
history = train_model(model, train_generator, validation_generator, epochs=10)

# ประเมินโมเดลกับชุดข้อมูลทดสอบ
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# บันทึกโมเดล
model.save('wildfire_detection_model.h5')
