# model_training.py
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def create_model():
    model = models.Sequential()

    # ใช้ 1x1 convolutions
    model.add(layers.Conv2D(64, (1, 1), activation=None, input_shape=(224, 224, 3)))  # ใช้ activation=None
    model.add(layers.BatchNormalization())  # เพิ่ม Batch Normalization
    model.add(layers.Activation('relu'))  # ใช้ Activation Layer แยก
    model.add(layers.MaxPooling2D((2, 2)))

    # ใช้ 3x3 convolutions
    model.add(layers.Conv2D(64, (3, 3), activation=None))  # ใช้ activation=None
    model.add(layers.BatchNormalization())  # เพิ่ม Batch Normalization
    model.add(layers.Activation('relu'))  # ใช้ Activation Layer แยก
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation=None))  # ใช้ activation=None
    model.add(layers.BatchNormalization())  # เพิ่ม Batch Normalization
    model.add(layers.Activation('relu'))  # ใช้ Activation Layer แยก
    model.add(layers.MaxPooling2D((2, 2)))

    # ใช้ 5x5 convolutions
    model.add(layers.Conv2D(64, (5, 5), activation=None))  # ใช้ activation=None
    model.add(layers.BatchNormalization())  # เพิ่ม Batch Normalization
    model.add(layers.Activation('relu'))  # ใช้ Activation Layer แยก
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation=None))  # ใช้ activation=None
    model.add(layers.BatchNormalization())  # เพิ่ม Batch Normalization
    model.add(layers.Activation('relu'))  # ใช้ Activation Layer แยก
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))

    # ตั้งค่า optimizer ที่ใช้ learning rate decay
    optimizer = Adam(learning_rate=0.001, decay=1e-6)

    model.compile(optimizer=optimizer,
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
