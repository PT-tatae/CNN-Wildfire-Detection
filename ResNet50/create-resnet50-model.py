import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'forest_fire/Training and Validation'
test_dir = 'forest_fire/Testing'

def load_data(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

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

def create_model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.Conv2D(64, (3, 3), activation='relu'),  # เพิ่ม Convolutional Layer
        layers.MaxPooling2D((2, 2)),  # เพิ่ม MaxPooling Layer
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),  # เพิ่ม Fully Connected Layer
        layers.Dropout(0.2), 
        layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, train_generator, validation_generator, epochs=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]  # Added ReduceLROnPlateau callback
    )
    return history

def plot_metrics(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# โหลดข้อมูล
train_generator, validation_generator, test_generator = load_data(train_dir, test_dir)

# สร้างโมเดล
model = create_model()

# ฝึกโมเดล
history = train_model(model, train_generator, validation_generator, epochs=15)

# ประเมินโมเดลกับชุดข้อมูลทดสอบ
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# บันทึกโมเดล
model.save('wildfire_detection_model_finetuned.keras')

# สร้างกราฟเปรียบเทียบ
plot_metrics(history)
