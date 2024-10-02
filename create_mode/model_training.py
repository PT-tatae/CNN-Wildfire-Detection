# model_training.py
from tensorflow.keras import layers, models

def create_model():
    # สร้างโมเดล CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # สำหรับ binary classification

    # คอมไพล์โมเดล
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, train_generator, validation_generator, epochs=10):
    # ฝึกโมเดล
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )
    return history
